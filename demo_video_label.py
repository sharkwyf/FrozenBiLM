import math
import sys
import os
import torch
import torch.nn
import torch.optim
import numpy as np
import random
import cv2
import kornia
import time
import json
import datetime
import argparse
from torch.utils.data import DataLoader, DistributedSampler
from collections import namedtuple
from queue import PriorityQueue, Queue
from pathlib import Path

from datasets import build_videotext_dataset, videotext_collate_fn, build_minedojo_videotext_dataset, minedojo_videotext_collate_fn
from model import build_model, get_tokenizer
from util.misc import get_mask, mask_tokens, adjust_learning_rate
from util import dist
from util.metrics import MetricLogger
from args import get_args_parser
from model.mineclip import MineCLIP, utils as U
from util.misc import get_mask, mask_tokens, adjust_learning_rate



MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)

"""
Label intentions given a video 
"""

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

@torch.no_grad()
def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Load mineclip model
    clip_param = {
        "arch": "vit_base_p16_fz.v2.t2",
        "hidden_dim": 512,
        "image_feature_dim": 512,
        "mlp_adapter_spec": "v0-2.t0",
        "pool_type": "attn.d2.nh8.glusw",
        "resolution": [160, 256],
    }
    clip_model = MineCLIP(**clip_param).to(device)
    clip_model.load_ckpt(args.model_path, strict=True)
    clip_model.clip_model.vision_model.projection = None

    # load frozenbilm model
    model = build_model(args)
    model.to(device)
    tokenizer = get_tokenizer(args)

    # Load pretrained checkpoint
    if args.load:
        print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)

    # Extract frames from video
    cap = cv2.VideoCapture(args.video_path)
    frames = []
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))
    size = (frame_width, frame_height)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    frames = np.stack(frames)
    cap.release()

    result = cv2.VideoWriter(f"{args.output_dir}{Path(args.video_path).stem}_{Path(args.load).stem}.avi",
        cv2.VideoWriter_fourcc(*'MJPG'),
        frame_rate, size)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Extract features
    # L, H, W, C -> L, C, H, W
    t_frames = torch.tensor(frames.transpose(0, 3, 1, 2), dtype=float, device=device)
    t_frames = resize_frames(t_frames, clip_param["resolution"])
    t_frames = torch_normalize(t_frames / 255.0, mean=MC_IMAGE_MEAN, std=MC_IMAGE_STD)
    # L, H, W, C -> L / fps, 768
    features = clip_model.forward_image_features(t_frames[::frame_rate]).unsqueeze(0)
    print("len of features: ", features.shape)

    texts = ["I was [MASK] [MASK]", "To craft a stone pickaxe, now I need to [MASK] [MASK] [MASK] [MASK]"]
    rt = [t for t in texts]
    for i in range(len(frames)):
        if (i - args.max_feats * frame_rate) >= 0 and (i - args.max_feats * frame_rate) % (args.sample_interval * frame_rate) == 0:
            print (i, "/", len(frames), "[", (i - args.max_feats * frame_rate) // frame_rate,  i // frame_rate, "]")
            video = features[:, (i - args.max_feats * frame_rate) // frame_rate: i // frame_rate]
            video_len = torch.tensor(video.size(1), device=device)
            video_mask = get_mask(video_len, video.size(1)).to(device)

            rt = [tokenizer(
                text,
                add_special_tokens=True,
                max_length=args.max_tokens,
                truncation=True,
                return_tensors="pt",
            ) for text in texts]
            for num, encoded in enumerate(rt):
                while True:
                    input_ids = encoded["input_ids"].to(device)
                    indices = ((input_ids[0] == 128000) * torch.arange(input_ids.shape[1])).nonzero()
                    if len(indices) == 0:
                        break
                    min_idx = indices[0]

                    # forward
                    output = model(
                        video=video,
                        video_mask=video_mask,
                        input_ids=input_ids,
                        attention_mask=encoded["attention_mask"].to(device),
                    )
                    encoded_output = output.logits[:, video_len:].argmax(dim=2)
                    input_ids[0][min_idx] = encoded_output[0][min_idx]
                rt[num] = tokenizer.batch_decode(encoded_output[:, 1:-1])[0]
                
        for j, text in enumerate(rt):
            cv2.putText(frames[i], 
                "{}".format(text), 
                (50, 50 + 30 * j), 
                font, 0.6, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
        result.write(frames[i])

    # release the cap object
    result.release()
    # close all windows
    cv2.destroyAllWindows()
    print("Succeed")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    parser.add_argument("--video_path", default="./data/Minedojo/demo.mp4", type=str)
    parser.add_argument("--output_dir", default="./data/Minedojo/", type=str)
    parser.add_argument("--model_path", default="./data/Minedojo/attn.pth", type=str)
    parser.add_argument("--sample_interval", default=5, type=int)
    parser.add_argument("--half_precision", type=bool, default=True, help="whether to output half precision float or not")
    args = parser.parse_args()
    args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
