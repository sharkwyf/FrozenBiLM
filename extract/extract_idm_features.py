import sys
import json
import os
import cv2
import numpy as np
import math
import argparse
import pickle
import multiprocessing as mp
import torch
from multiprocessing import Process, Lock, Queue, Pool
from tqdm import tqdm
import sys
workdir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, workdir)
from util.pertrel_oss_helper import init_clients
import traceback
from model.idm.inverse_dynamics_model import IDMAgent

"""
Load MP4 videos and extract idm features of 64 frames in 16s
"""
def init_idm_agent(model, weights, rank, n_gpu):
    """return n_process clip models evenly distributed on n_gpu gpus"""
    device = torch.device("cuda", rank % n_gpu) if n_gpu > 0 else "cpu"
    print("loading idm model on device", device)
    agent_parameters = pickle.load(open(model, "rb"))
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs, device=device)
    agent.load_weights(weights)
    return agent


def download(args, queue_videos, done, videos):
    """
    Fetch video clips of interval [-32 + start, 96 + end], 128 + 16 * fps frames in total
    name: vid_id
    """
    client = init_clients(1)[0]
    for vid_id, vid_clips in videos:
        try:
            # load video
            cap = client.load_video(f"{args.input_path}{vid_id}{args.vid_suffix}")
            fps, frame_count = cap.get(5), cap.get(7)
            skipped = 0
            
            # extract frames from vid_clips
            vid_clips = sorted(list(vid_clips))
            frames_dict = {}
            clips_dict = {}
            for i, word_start in enumerate(sorted(list(vid_clips))):
                # extract frames
                f_start = int(fps * (word_start + args.vid_start)) - 32
                f_end = int(fps * (word_start + args.vid_end)) + 96
                if f_start < 0 or f_end > frame_count - 1:
                    done.put(1)
                    skipped += 1
                    continue
                indices = np.linspace(f_start + 32, f_end - 96, num=args.n_frames)
                indices = np.array(list(map(round, indices)))
                for index in range(f_start, f_end):
                    frames_dict[index] = None
                clips_dict[word_start] = (f_start, f_end, indices)

            # if skipped > 0:
            #     print(f"skipped {skipped} clips")
            frames_keys = list(frames_dict.keys())
            if len(frames_keys) == 0:
                continue
            min_frame, max_frame = min(frames_keys), max(frames_keys)
            for i in range(0, max_frame + 1):
                if i in frames_dict:
                    ret, frame = cap.read()
                    # BGR -> RGB
                    frame = frame[..., ::-1]
                    frames_dict[i] = cv2.resize(frame, args.resolution)
                else:
                    cap.grab()

            queue_videos.put([vid_id, frames_dict, clips_dict])
        except Exception:
            print(traceback.format_exc())
            print(sys.exc_info()[2])


@torch.no_grad()
def extract(args, queue_videos, queue_features, lock, rank):
    """Extract features using IDM"""
    device = torch.device("cuda", rank % args.n_gpu) if args.n_gpu > 0 else "cpu"
    agent = init_idm_agent(args.model, args.weights, rank, args.n_gpu)
    while True:
        try:
            [vid_id, frames_dict, clips_dict] = queue_videos.get()
            for word_start in clips_dict.keys():
                rs = []
                agent.reset()
                f_start, f_end, indices = clips_dict[word_start]
                for i in range(f_start, f_end, 64):
                    agent.reset()
                    low, high = i, i + 128
                    if high > f_end:
                        break
                    # N, H, W, C
                    frames = np.array([frames_dict[j] for j in range(low, high)])
                    # N, 4096
                    predicted_actions, pi_h = agent.predict_actions(frames)
                    rs.append(pi_h[0, 32:96])
                idm_feats = torch.concat(rs)
                idm_feats = idm_feats[indices - f_start - 32]
                
                queue_features.put([f"{vid_id}_{word_start}", idm_feats])

        except Exception:
            print(traceback.format_exc())
            print(sys.exc_info()[2])

def upload(args, queue_features, done, lock):
    """Save feats and words to Ceph"""
    client = init_clients(1)[0]
    while True:
        try:
            [name, output] = queue_features.get()
            client.save_nbz(f"{args.output_path}{name}{args.feats_suffix}", output.cpu().numpy())
            done.put(1)
        except Exception:
            print(traceback.format_exc())
            print(sys.exc_info()[2])

def update(current_to_process, done):
    """Update tqdm bar"""
    pbar = tqdm(total=current_to_process)
    processed_cnt = 0
    while processed_cnt < current_to_process:
        n = done.get()
        pbar.update(n)
        processed_cnt += n


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--video_index_file', type=str, default=os.path.join(workdir, 'data/Minedojo/minedojo_clips_test.json'))
    parser.add_argument('--input_path', type=str, default='s3://minedojo/videos_test/')
    parser.add_argument('--output_path', type=str, default='s3://minedojo/idms/test/')
    parser.add_argument('--vid_suffix', type=str, default=".mp4")
    parser.add_argument('--feats_suffix', type=str, default=".nbz")
    # idm
    parser.add_argument("--weights", type=str, default=os.path.join(workdir, "data/Minedojo/4x_idm.weights"), help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, default=os.path.join(workdir, "data/Minedojo/4x_idm.model"), help="Path to the '.model' file to be loaded.")
    # video
    parser.add_argument('--vid_start', type=float, default=-8, help="start of video clip ")
    parser.add_argument('--vid_end', type=float, default=8)
    parser.add_argument('--n_frames', type=int, default=64)
    parser.add_argument('--resolution', type=int, nargs="+", default=[128, 128])
    parser.add_argument('--max_clips_per_keyword', type=int, default=math.inf)
    # multi machines
    parser.add_argument("--n_downloader", default=1, type=int)
    parser.add_argument("--n_extractor", default=1, type=int)
    parser.add_argument("--n_uploader", default=1, type=int)
    parser.add_argument("--n_gpu", default=torch.cuda.device_count(), type=int)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    print(args)
    print(f"total cpu counts: {mp.cpu_count()}")
    mp.set_start_method('spawn', force = True)

    with open(args.video_index_file) as f:
        video_indices = json.load(f)
    
    client = init_clients(1)[0]
    print("fetching indices")
    files = set(client.list(args.input_path))
    print(f"loaded {len(files)} files")

    print("fetching processed indices")
    processed_indices = set([x[:-4] for x in client.list(args.output_path)])
    print(f"loaded {len(processed_indices)} processed indices")

    videos = {}
    for keyword, items in video_indices.items():
        if keyword in []:
            pass
        np.random.seed(43)
        np.random.shuffle(items)
        total_len = min(args.max_clips_per_keyword, len(items))
        for vid_id, word_start, word_len in items[:total_len]:
            if f"{vid_id}{args.vid_suffix}" in files :
                if f"{vid_id}_{word_start:02}" not in processed_indices:
                    if vid_id not in videos:
                        videos[vid_id] = set()
                    videos[vid_id].add(word_start)

    inputs = sorted([(k, v) for k, v in videos.items()])
    total_to_process = sum([len(v) for k, v in inputs])
    print(f"max clips per keyword: {args.max_clips_per_keyword}, total clips: {total_to_process}")

    low = args.rank * len(inputs) // args.world_size
    high = (args.rank + 1) * len(inputs) // args.world_size
    inputs = inputs[low:high]
    current_to_process = sum([len(v) for k, v in inputs])
    print(f"rank: {args.rank + 1}/{args.world_size}, current clips to process: {current_to_process}")

    # Extract IDM features
    lock = Lock()
    queue_videos, queue_features, done = Queue(1024), Queue(1024), Queue(1024)
    downloaders, extractors, uploaders = [], [], []
    for n in range(args.n_downloader):
        downloaders.append(Process(target=download, args=(args, queue_videos, done, [x for x in inputs[n::args.n_downloader]])))
    for n in range(args.n_extractor):
        p = Process(target=extract, args=(args, queue_videos, queue_features, lock, n))
        p.daemon = True
        extractors.append(p)
    for n in range(args.n_uploader):
        p = Process(target=upload, args=(args, queue_features, done, lock))
        p.daemon = True
        uploaders.append(p)

    p0 = Process(target=update, args=(current_to_process, done)).start()
    for c in extractors:
        c.start()
    for c in uploaders:
        c.start()
    for p in downloaders:
        p.start()

    print("all prcoesses started")

    for p in downloaders:
        p.join()

    print("processed video clips", sum([current_to_process]))