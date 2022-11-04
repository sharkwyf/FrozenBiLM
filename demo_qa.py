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


"""
Label intentions given a video 
"""
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

    # encoded available words
    answer_id = tokenizer.encode(["▁" + w for w in list(ALL_WORDS)])[1:-1]
    answer_bias = torch.zeros(tokenizer.vocab_size, dtype=torch.float32, device=device)
    for id in answer_id:
        answer_bias[id] = args.answer_bias_weight

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
    resized = [256, 160]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, resized))
    frames = np.stack(frames)
    cap.release()

    # Extract features
    # L, H, W, C -> L, C, H, W
    t_frames = U.any_to_torch_tensor(frames.transpose(0, 3, 1, 2), dtype=torch.float, device=device)
    # L, H, W, C -> L / fps, 768
    features = clip_model.forward_image_features(t_frames[::frame_rate]).cpu().numpy()
    features = torch.from_numpy(features).float().unsqueeze(0)
    print("len of features: ", features.shape, "dtype: ", features.dtype)

    video = features[:, args.start:args.end]
    video = video[:, ::1]
    video_len = torch.tensor(video.size(1), device=device)
    video_mask = get_mask(video_len, video.size(1)).to(device)
    print("input video len:", video_len, "frame rate:", frame_rate)
    while True:
        text = input("Please input: ").lower().replace(",", " ").replace(".", " ").replace("[mask]", "[MASK]")

        # command
        if text == "switch":
            args.answer_bias_weight = 100 - args.answer_bias_weight
            answer_bias = torch.zeros(tokenizer.vocab_size, dtype=torch.float32, device=device)
            for id in answer_id:
                answer_bias[id] = args.answer_bias_weight
            print("switched", args.answer_bias_weight)
            continue
        if text.startswith("interval"):
            args.start = int(text.split()[1])
            args.end = int(text.split()[2])
            print("start", args.start, "end", args.end)
            video = features[:, args.start:args.end]
            video = video[:, ::1]
            video_len = torch.tensor(video.size(1), device=device)
            video_mask = get_mask(video_len, video.size(1)).to(device)
            print("input video len:", video_len, "frame rate:", frame_rate)

            result = cv2.VideoWriter(f"{args.output_dir}qa_{Path(args.video_path).stem}.avi",
                cv2.VideoWriter_fourcc(*'MJPG'),
                1, resized)
            for fr in frames[::frame_rate][args.start:args.end]:
                result.write(fr)
            result.release()
            continue

        rt = [tokenizer(
            text,
            add_special_tokens=True,
            max_length=args.max_tokens,
            truncation=True,
            return_tensors="pt",
        )]
        for num, encoded in enumerate(rt):
            while True:
                input_ids = encoded["input_ids"].to(device)
                indices = ((input_ids[0] == tokenizer.mask_token_id) * torch.arange(input_ids.shape[1], device=device)).nonzero()
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
                logits = output.logits[:,video_len:,:len(answer_bias)] + answer_bias
                encoded_output = logits.argmax(dim=2)

                # generate one word at a time
                input_ids[0][min_idx] = encoded_output[0][min_idx]
            rt[num] = tokenizer.batch_decode(input_ids[:, 1:-1])[0]
        print("Output:", rt[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    parser.add_argument("--video_path", default="./data/Minedojo/animals.mp4", type=str)
    parser.add_argument("--output_dir", default="./data/Minedojo/", type=str)
    parser.add_argument("--model_path", default="./data/Minedojo/attn.pth", type=str)
    parser.add_argument("--answer_bias_weight", default=0, type=float)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=160, type=int)
    args = parser.parse_args()
    args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
