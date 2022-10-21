

import torch
import os
import numpy as np
from io import StringIO, BytesIO
import kornia
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import argparse
import sys
workdir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, workdir)
from model.mineclip import MineCLIP
from petrel_client.client import Client


MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)

"""
Transfer video frames into Vit features
then, split captions into past, current, future sentences
"""
def get_args():
    parser = argparse.ArgumentParser(description="Easy video feature extractor")
    parser.add_argument("--trans", default="s3://minedojo/trans/v1/", type=str)
    parser.add_argument("--output_path", default="./data/Minedojo/mineclip_features.npy", type=str)
    parser.add_argument("--model_path", default="./data/Minedojo/attn.pth", type=str)
    parser.add_argument("--cluster", default="cluster1", type=str)
    parser.add_argument("--keywords", default=[], type=str, nargs="+")
    parser.add_argument("--world_size", default=8, type=int)
    parser.add_argument("--half_precision", type=bool, default=True, help="whether to output half precision float or not")
    args = parser.parse_args()
    return args

def torch_normalize(tensor: torch.Tensor, mean, std, inplace=False):
    """
    Adapted from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#normalize

    Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("tensor should be a torch tensor. Got {}.".format(type(tensor)))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            f"std evaluated to zero after conversion to {dtype}, leading to division by zero."
        )
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.sub_(mean).div_(std)
    return tensor

def resize_frames(frames, resolution):
    return kornia.geometry.transform.resize(frames, resolution).clamp(0.0, 255.0)

def run_extract_features(rank, world_size, args, clips, mp_lock):
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if torch.cuda.is_available() else "cpu"

    clip_param = {
        "arch": "vit_base_p16_fz.v2.t2",
        "hidden_dim": 512,
        "image_feature_dim": 512,
        "mlp_adapter_spec": "v0-2.t0",
        "pool_type": "attn.d2.nh8.glusw",
        "resolution": [160, 256],
    }
    model = MineCLIP(**clip_param).to(device)
    model.load_ckpt(args.model_path, strict=True)
    model.clip_model.vision_model.projection = None

    client = Client()
    pbar = tqdm(total=sum([len(clips[keyword][rank::world_size]) for keyword in args.keywords]))
    with torch.no_grad():
        output_features = {}
        for keyword in args.keywords:
            output_features[keyword] = {}
            for i, p in enumerate(clips[keyword][rank::world_size]):
                # Extract video features
                stream = BytesIO(client.get(f"{args.trans}{keyword}/{p}.npy"))   # enable_stream=True for large data
                frames = np.load(stream, encoding="bytes")
                # 64, H, W, C -> 64, C, H, W
                frames = torch.tensor(frames[::].transpose(0, 3, 1, 2), dtype=float, device=device)
                frames = resize_frames(frames, clip_param["resolution"])
                frames = torch_normalize(frames / 255.0, mean=MC_IMAGE_MEAN, std=MC_IMAGE_STD)
                features = model.forward_image_features(frames).cpu().numpy()
                if args.half_precision:
                    features = features.astype("float16")

                # Extract captions
                stream = BytesIO(client.get(f"{args.trans}{keyword}/{p}.txt"))  # enable_stream=True for large data
                text = stream.read().decode("utf-8")
                splits = [item.split("|") for item in text.replace(" ", "").split("\n")]
                err = [x for x in splits[:-1] if len(x) != 3]
                if len(err) > 0:
                    print(err)
                splits = [x for x in splits if len(x) == 3]
                captions = {
                    "start": np.array([float(x[0])for x in splits], dtype=np.float16),
                    "end": np.array([float(x[1])for x in splits], dtype=np.float16),
                    "word": np.array([x[2]for x in splits]),
                }

                output_features[keyword][p] = [features, captions]
                pbar.update(1)
                # break
        
        # Gather features and save to npy
        print(dist.get_rank(), output_features)
        if dist.get_rank() == 0:
            rt = [None for _ in range(dist.get_world_size())]
            dist.gather_object(output_features, rt, dst=0)
            for i in range(1, len(rt)):
                for keyword in args.keywords:
                    output_features[keyword].update(rt[i][keyword])
            np.save(args.output_path, output_features)
            print(f"Extraction completed, saved to {args.output_path}, total length: {sum([len(output_features[keyword]) for keyword in args.keywords])}")
        else:
            dist.gather_object(output_features, dst=0)


def main(args):
    # mp_array = mp.Array("i", [0 for _ in range(cfg.world_size)])
    mp_lock = mp.Lock()
    client = Client()
    if len(args.keywords) == 0:
        args.keywords = [x[:-1] for x in list(client.list(f"{args.cluster}:{args.trans}"))]
    print("keywords: ", args.keywords)
    clips = {
        keyword: [
            x[:-4] for x in list(client.list(f"{args.cluster}:{args.trans}{keyword}/")) if x.endswith(".npy")
        ] for keyword in args.keywords
    }
    print("total clips:", sum([len(v)for v in clips.values()]))
    mp.spawn(run_extract_features,
        args=(args.world_size, args, clips, mp_lock),
        nprocs=args.world_size,
        join=True)

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(49505)
    mp.set_start_method('spawn', force = True)
    args = get_args()
    print(args)
    main(args)