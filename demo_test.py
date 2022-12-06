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
from util.verb_noun import VERB_NOUN_PAIRS, VERB_PHASE, ALL_WORDS
from util.pertrel_oss_helper import init_clients


"""
Label intentions given a video 
"""
def resize_frames(frames, resolution):
    return kornia.geometry.transform.resize(frames, resolution).clamp(0.0, 255.0)

@torch.no_grad()
def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    client = init_clients(1)[0]
    video_name = "-1ZL1W8bF6A"
    word_start = 1651.44
    video_path = f"s3://minedojo/videos/{video_name}.mp4"
    clip_path = f"s3://minedojo/trans/v1/{video_name}_{word_start}.nbz"
    feat_path = f"s3://minedojo/feats/v2/{video_name}_{word_start}.npz"

    cap = client.load_video(video_path)
    clips = client.load_nbz(clip_path)
    label_feats = client.load_npz(feat_path).item()["feats"]

    resized = [256, 160]
    fps, frame_count = cap.get(5), cap.get(7)
    frames_dict = {}
    clips_dict = {}
    f_start = max(0, int(fps * (word_start -8)))
    f_end = min(frame_count - 1, int(fps * (word_start + 8)))
    indices = np.linspace(f_start, f_end, num=64)
    indices = list(map(round, indices))
    for index in indices:
        frames_dict[index] = None
    clips_dict[word_start] = indices

    frames_keys = list(frames_dict.keys())
    min_frame, max_frame = min(frames_keys), max(frames_keys)
    for i in range(0, max_frame + 1):
        if i in frames_dict:
            ret, frame = cap.read()
            # BGR -> RGB
            frame = frame[..., ::-1]
            frames_dict[i] = cv2.resize(frame, resized)
        else:
            cap.grab()

    frames = []
    for index in clips_dict[word_start]:
        frames.append(frames_dict[index])
    frames = np.stack(frames)


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
    clip_model.eval()

    # load frozenbilm model
    model = build_model(args)
    model.to(device)
    model.eval()
    tokenizer = get_tokenizer(args)

    # Load pretrained checkpoint
    if args.load:
        print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)

    result = cv2.VideoWriter(f"{args.output_dir}{Path(args.video_path).stem}_{Path(args.load).stem}_w{args.answer_bias_weight}.avi",
        cv2.VideoWriter_fourcc(*'MJPG'),
        4, resized)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Extract features
    # L, H, W, C -> L, C, H, W
    t_frames = torch.tensor(frames.transpose(0, 3, 1, 2), dtype=torch.float, device=device)
    # L, H, W, C -> L / fps, 768
    features = clip_model.forward_image_features(t_frames[::1]).cpu().numpy()
    features = torch.from_numpy(features).float().unsqueeze(0)

    print("len of features: ", features.shape, "dtype: ", features.dtype)

    for i in range(len(frames)):
        result.write(frames[i])

    # release the cap object
    result.release()
    # close all windows
    cv2.destroyAllWindows()
    print("Succeed")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    parser.add_argument("--video_path", default="./data/Minedojo/animals.mp4", type=str)
    parser.add_argument("--output_dir", default="./data/Minedojo/", type=str)
    parser.add_argument("--model_path", default="./data/Minedojo/attn.pth", type=str)
    parser.add_argument("--answer_bias_weight", default=100, type=float)
    args = parser.parse_args()
    args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
