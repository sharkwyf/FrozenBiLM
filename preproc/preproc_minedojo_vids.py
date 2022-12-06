import sys
import json
import os
import cv2
import numpy as np
import bisect
import copy
import time
import math
import argparse
import multiprocessing as mp
from multiprocessing import Process, Lock, Queue, Pool
from tqdm import *
from petrel_client.client import Client
import sys
workdir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, workdir)
from util.pertrel_oss_helper import init_clients
import traceback

"""
Load MP4 videos and extract 64 frames & captions in 64s
"""
def get_word_list(vtt):
    """Assign start time and length to every word in vtt"""
    word_list = []
    for caption in vtt.captions:
        text = caption.text.split()
        t_start = caption.start_in_seconds
        t_len = (caption.end_in_seconds - caption.start_in_seconds) / len(text)
        for i, word in enumerate(text):
            word_list.append((word, t_start + i * t_len, t_len))
    return word_list

def format_word_list(word_list, t_start, t_end, keyword_start):
    """Return formatted [start]|[len]|[word] in text"""
    words = [word for word, _t_start, _t_len in word_list]
    starts = [_t_start for word, _t_start, _t_len in word_list]
    start_index = max(bisect.bisect_left(starts, t_start), 0)
    end_index = min(bisect.bisect_left(starts, t_end), len(words))
    assert start_index <= end_index, f'{start_index},{end_index},{t_start},{t_end}'
    ret = ''.join([f'{round(word_list[i][1]-keyword_start, 2)}|{round(word_list[i][2], 2)}|{word_list[i][0]}\n' for i in range(start_index, end_index)])
    return ret


def preprocess_minedojo_videotext_data(intput):
    try:
        client = clients[mp.current_process()._identity[0] - 1]
        args, vid_id, vid_clips = intput
        processed_cnt = 0

        # load video
        cap = client.load_video(f"{args.input_path}{vid_id}{args.vid_suffix}")
        fps, frame_count = cap.get(5), cap.get(7)

        # load vtt
        vtt = client.load_vtt(f"{args.input_path}{vid_id}{args.vtt_suffix}")
        word_list = get_word_list(vtt)

        # extract frames and captions from vid_clips
        vid_clips = sorted(list(vid_clips))
        frames_dict = {}
        clips_dict = {}
        for i, word_start in enumerate(sorted(list(vid_clips))):
            # extract frames
            f_start = max(0, int(fps * (word_start + args.vid_start)))
            f_end = min(frame_count - 1, int(fps * (word_start + args.vid_end)))
            indices = np.linspace(f_start, f_end, num=args.n_frames)
            indices = list(map(round, indices))
            for index in indices:
                frames_dict[index] = None
            clips_dict[word_start] = indices
        
        frames_keys = list(frames_dict.keys())
        min_frame, max_frame = min(frames_keys), max(frames_keys)
        # cap.set(1, min_frame)
        for i in range(0, max_frame + 1):
            if i in frames_dict:
                ret, frame = cap.read()
                # BGR -> RGB
                frame = frame[..., ::-1]
                frame = cv2.resize(frame, args.resolution)
                frames_dict[i] = frame
            else:
                cap.grab()

        for i, word_start in enumerate(sorted(list(vid_clips))):
            # extract frames
            try:
                frames = []
                for index in clips_dict[word_start]:
                    frames.append(frames_dict[index])
                frames = np.stack(frames)

                # extract txt
                data_txt = format_word_list(word_list, word_start + args.cap_start, word_start + args.cap_end, word_start)

                # save files
                client.save_nbz(f"{args.output_path}{vid_id}_{word_start}.nbz", frames)
                client.save_txt(f"{args.output_path}{vid_id}_{word_start}.txt", data_txt) 
                processed_cnt += 1
            except:
                print(f"{vid_id}_{word_start} processed failed")
                print(traceback.format_exc())
                print(sys.exc_info()[2])
                continue
        cap.release()
    except Exception:
        print(traceback.format_exc())
        print(sys.exc_info()[2])
        return 0
    return processed_cnt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--video_index_file', type=str, default='./data/Minedojo/minedojo_clips.json')
    parser.add_argument('--input_path', type=str, default='s3://minedojo/videos_test/')
    parser.add_argument('--output_path', type=str, default='s3://minedojo/trans/test/')
    parser.add_argument('--vid_suffix', type=str, default=".mp4")
    parser.add_argument('--vtt_suffix', type=str, default=".en.vtt")
    parser.add_argument('--n_process', type=int, default=mp.cpu_count())
    # video
    parser.add_argument('--vid_start', type=float, default=-8, help="start of video clip ")
    parser.add_argument('--vid_end', type=float, default=8)
    parser.add_argument('--n_frames', type=int, default=64)
    parser.add_argument('--resolution', type=int, nargs="+", default=[256, 160])
    parser.add_argument('--max_clips_per_keyword', type=int, default=math.inf)
    # caption
    parser.add_argument('--cap_start', type=float, default=-40)
    parser.add_argument('--cap_end', type=float, default=24)
    # multi machines
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    print(args)
    print(f"total cpu counts: {mp.cpu_count()}")

    with open(args.video_index_file) as f:
        video_indices = json.load(f)
    
    clients = init_clients(args.n_process)
    print("fetching indices")
    files = set(clients[0].list(args.input_path))
    print(f"loaded {len(files)} files")

    print("fetching downloaded indices")
    downloaded_indices = set([x[:-4] for x in clients[0].list(args.output_path)])
    print(f"loaded {len(downloaded_indices)} downloaded indices")

    videos = {}
    for keyword, items in video_indices.items():
        if keyword in []:
            pass
        np.random.seed(43)
        np.random.shuffle(items)
        total_len = min(args.max_clips_per_keyword, len(items))
        for vid_id, word_start, word_len in items[:total_len]:
            if f"{vid_id}{args.vid_suffix}" in files and f"{vid_id}{args.vtt_suffix}" in files:
                if f"{vid_id}_{word_start:02}" not in downloaded_indices:
                    if vid_id not in videos:
                        videos[vid_id] = set()
                    videos[vid_id].add(word_start)

    inputs = sorted([(args, k, v) for k, v in videos.items()])
    low = args.rank * len(inputs) // args.world_size
    high = (args.rank + 1) * len(inputs) // args.world_size
    print(f"max clips per keyword: {args.max_clips_per_keyword}, total clips: {sum([len(v) for _, _, v in inputs])}")
    inputs = inputs[low:high]
    print(f"rank: {args.rank + 1}/{args.world_size}, current clips: {sum([len(v) for _, _, v in inputs])}")

    with Pool(processes=args.n_process) as pool:
        results = list(tqdm(
            pool.imap_unordered(
                preprocess_minedojo_videotext_data,
                inputs,
            ),
            total=len(inputs)
        ))
    print("total processed video clips", sum(results))