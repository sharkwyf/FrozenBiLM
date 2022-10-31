

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
def init_clip_models(n_process, n_gpu):
    """return n_process clip models evenly distributed on n_gpu gpus"""
    clip_param = {
        "arch": "vit_base_p16_fz.v2.t2",
        "hidden_dim": 512,
        "image_feature_dim": 512,
        "mlp_adapter_spec": "v0-2.t0",
        "pool_type": "attn.d2.nh8.glusw",
        "resolution": [160, 256],
    }
    models = []
    for rank in range(n_process):
        device = torch.device("cuda", rank % n_gpu) if n_gpu > 0 else "cpu"
        model = MineCLIP(**clip_param).to(device)
        model.load_ckpt(args.model_path, strict=True)
        model.clip_model.vision_model.projection = None
        model.eval()
        models.append(model)
    for model in models:
        print(next(model.parameters()).device)
    return models

@torch.no_grad()
def run_extract_features(input):
    try:
        args, name, clients, models = input
        rank = mp.current_process()._identity[0] - 1
        client = clients[rank]
        model = models[rank]
        device = torch.device("cuda", rank % args.n_gpu) if args.n_gpu > 0 else "cpu"

        # Extract video features
        frames = client.load_nbz(f"{args.input_path}{name}.nbz")
        # 64, H, W, C -> 64, C, H, W
        frames = U.any_to_torch_tensor(frames.transpose(0, 3, 1, 2), dtype=torch.uint8, device=device)
        image_feats = model.forward_image_features(frames).cpu().numpy()
        if args.half_precision:
            image_feats = image_feats.astype("float16")

        # Extract captions
        text = client.load_txt(f"{args.input_path}{name}.txt")
        splits = [item.split("|") for item in text.replace(" ", "").split("\n")]
        err = [x for x in splits[:-1] if len(x) != 3]
        if len(err) > 0:
            print(err)
        splits = [x for x in splits if len(x) == 3]
        output = {
            "feats": image_feats,
            "words": np.array([x[2]for x in splits]),
            "starts": np.array([float(x[0])for x in splits], dtype=np.float16),
            "lens": np.array([float(x[1])for x in splits], dtype=np.float16),
        }
        client.save_npz(f"{args.output_path}{name}.npz", output)
    except Exception:
        print(traceback.format_exc())
        print(sys.exc_info()[2])
        return 0
    return 1


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(49600)
    mp.set_start_method('spawn', force = True)
    
    parser = argparse.ArgumentParser(description="Easy video feature extractor")
    parser.add_argument("--input_path", default="s3://minedojo/trans/v1/", type=str)
    parser.add_argument("--output_path", default="s3://minedojo/feats/v1/", type=str)
    parser.add_argument("--model_path", default="./data/Minedojo/attn.pth", type=str)
    parser.add_argument("--n_process", default=1, type=int)
    parser.add_argument("--n_gpu", default=torch.cuda.device_count(), type=int)
    parser.add_argument("--half_precision", type=bool, default=True, help="whether to output half precision float or not")
    args = parser.parse_args()
    print(args)

    clients = init_clients(args.n_process)
    models = init_clip_models(args.n_process, args.n_gpu)

    # load clip indices & models
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
            indices.append([args, file[:-4], clients, models])
    print(f"total clips to extract: {len(indices)}")
    with Pool(processes=args.n_process) as pool:
        results = list(tqdm(
            pool.imap_unordered(
                run_extract_features,
                indices,
            ),
            total=len(indices)
        ))
    print("total processed video clips", sum(results))

