

import torch
import os
import sys
import traceback
import numpy as np
import argparse
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Lock, Queue, Pool
from tqdm import tqdm
workdir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, workdir)
from model.mineclip import MineCLIP, utils as U
from util.pertrel_oss_helper import init_clients

"""
Transfer video frames into Vit features
then, split captions into past, current, future sentences
"""
def init_clip_models(args, rank, n_gpu):
    """return n_process clip models evenly distributed on n_gpu gpus"""
    clip_param = {
        "arch": "vit_base_p16_fz.v2.t2",
        "hidden_dim": 512,
        "image_feature_dim": 512,
        "mlp_adapter_spec": "v0-2.t0",
        "pool_type": "attn.d2.nh8.glusw",
        "resolution": [160, 256],
    }
    device = torch.device("cuda", rank % n_gpu) if n_gpu > 0 else "cpu"
    model = MineCLIP(**clip_param).to(device)
    model.load_ckpt(args.model_path, strict=True)
    model.clip_model.vision_model.projection = None
    model.eval()
    print(next(model.parameters()).device)
    return model


def producer(args, queue1, lock, names):
    """Fetch video clips and captions"""
    client = init_clients(1)[0]
    for name in names:
        try:
            # Load video features
            frames = client.load_nbz(f"{args.input_path}{name}.nbz")

            # Load captions
            text = client.load_txt(f"{args.input_path}{name}.txt")
            splits = [item.split("|") for item in text.replace(" ", "").split("\n")]
            splits = [x for x in splits if len(x) == 3]

            queue1.put([name, frames, splits])
        except Exception:
            print(traceback.format_exc())
            print(sys.exc_info()[2])


@torch.no_grad()
def consumer1(args, queue1, queue2, lock, rank):
    """Extract features using ViT"""
    device = torch.device("cuda", rank % args.n_gpu) if args.n_gpu > 0 else "cpu"
    model = init_clip_models(args, rank, args.n_gpu)
    while True:
        try:
            [name, frames, splits] = queue1.get()
            # 64, H, W, C -> 64, C, H, W
            frames = U.any_to_torch_tensor(frames.transpose(0, 3, 1, 2), dtype=torch.float, device=device)
            image_feats = model.forward_image_features(frames).cpu().numpy()

            output = {
                "feats": image_feats,
                "words": np.array([x[2]for x in splits]),
                "starts": np.array([float(x[0])for x in splits], dtype=np.float16),
                "lens": np.array([float(x[1])for x in splits], dtype=np.float16),
            }

            queue2.put([name, output])
        except Exception:
            print(traceback.format_exc())
            print(sys.exc_info()[2])

def consumer2(args, queue2, done, lock):
    """Save feats and words to Ceph"""
    client = init_clients(1)[0]
    while True:
        try:
            [name, output] = queue2.get()
            client.save_npz(f"{args.output_path}{name}.npz", output)
            done.put(1)
        except Exception:
            print(traceback.format_exc())
            print(sys.exc_info()[2])


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(49600)
    mp.set_start_method('spawn', force = True)
    
    parser = argparse.ArgumentParser(description="Easy video feature extractor")
    parser.add_argument("--input_path", default="s3://minedojo/trans/v1/", type=str)
    parser.add_argument("--output_path", default="s3://minedojo/feats/test/", type=str)
    parser.add_argument("--model_path", default="./data/Minedojo/attn.pth", type=str)
    parser.add_argument("--n_producer1", default=1, type=int)
    parser.add_argument("--n_consumer1", default=1, type=int)
    parser.add_argument("--n_consumer2", default=1, type=int)
    parser.add_argument("--n_gpu", default=torch.cuda.device_count(), type=int)
    args = parser.parse_args()
    print(args)

    # load clip indices & models
    clients = init_clients(1)
    print("fetching nbz indices")
    files = set(clients[0].list(args.input_path))
    # files = set(["-7OrkmVmS38_199.09.nbz", "-7OrkmVmS38_199.09.txt", "-7NkbSPZMLY_872.14.nbz", "-7NkbSPZMLY_872.14.txt"])
    print(f"loaded {len(files)} files")

    print("fetching extracted indices")
    downloaded_indices = set([x[:-4] for x in clients[0].list(args.output_path)])
    # downloaded_indices = set()
    print(f"loaded {len(downloaded_indices)} downloaded indices")

    # load clip indices to download
    indices = []
    for file in files:
        if not file.endswith(".nbz"):
            continue
        if f"{file[:-4]}.txt" in files and file[:-4] not in downloaded_indices:
            indices.append([args, file[:-4]])
    print(f"total clips to extract: {len(indices)}")

    pbar = tqdm(total=len(indices))
    lock = Lock()
    queue1, queue2, done = Queue(1024), Queue(1024), Queue(1024)
    producers, consumer1s, consumer2s = [], [], []
    for n in range(args.n_producer1):
        producers.append(Process(target=producer, args=(args, queue1, lock, [x[1] for x in indices[n::args.n_producer1]])))
    for n in range(args.n_consumer1):
        p = Process(target=consumer1, args=(args, queue1, queue2, lock, n))
        p.daemon = True
        consumer1s.append(p)
    for n in range(args.n_consumer2):
        p = Process(target=consumer2, args=(args, queue2, done, lock))
        p.daemon = True
        consumer2s.append(p)

    for p in producers:
        p.start()
    for c in consumer1s:
        c.start()
    for c in consumer2s:
        c.start()

    step = 0
    while True:
        n = done.get()
        pbar.update(n)
        step += 1
        if step == len(indices):
            break

    for p in producers:
        p.join()

    print("Finished")


