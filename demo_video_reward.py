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

from tqdm import tqdm
from datasets import build_videotext_dataset, videotext_collate_fn, build_minedojo_videotext_dataset, minedojo_videotext_collate_fn
from model import build_model, get_tokenizer
from util.misc import get_mask, mask_tokens, adjust_learning_rate
from util import dist
from util.metrics import MetricLogger
from args import get_args_parser
from model.mineclip import MineCLIP, utils as U
from util.misc import get_mask, mask_tokens, adjust_learning_rate
from util.verb_noun import VERB_NOUN_PAIRS, VERB_PHASE, ALL_WORDS, ALL_ACTIONS


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
    answer_id = tokenizer.encode(["▁" + w for w in list(ALL_ACTIONS)])[1:-1]
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
    resized_frames = []
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))
    size = (frame_width, frame_height)
    resized = [256, 160]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        resized_frames.append(cv2.resize(frame, resized))
    frames = np.stack(frames)
    resized_frames = np.stack(resized_frames)
    cap.release()

    result = cv2.VideoWriter(f"{args.output_dir}rew_{Path(args.video_path).stem}_{Path(args.load).stem}_{args.stack}_t{args.n_frames // args.frames_per_second}_b{args.answer_bias_weight}.avi",
        cv2.VideoWriter_fourcc(*'MJPG'),
        frame_rate, size)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Extract features
    # L, H, W, C -> L, C, H, W
    t_frames = U.any_to_torch_tensor(resized_frames.transpose(0, 3, 1, 2), dtype=torch.float, device=device)
    # L, H, W, C -> L // sample_rate, 768
    sample_rate = frame_rate // args.frames_per_second
    features = clip_model.forward_image_features(t_frames[::sample_rate]).cpu().numpy()
    features = torch.from_numpy(features).float().unsqueeze(0)
    print("len of features: ", features.shape, "dtype: ", features.dtype)

    ["move", "jump", "swim", "climb", "stand"]
    texts = [
        ("move", "i [mask]", "move"),
        ("jump", "i [mask]", "jump"),
        ("swim", "i [mask]", "swim"),
        ("climb", "i [mask]", "climb"),
        ("stand", "i [mask]", "stand"),
        
        # ("placed water", "i placed a [mask] block", "water"),
        # ("placed stone", "i placed a [mask] block", "stone"),
        # ("moving forward", "i'm moving [mask]", "forward"),
        # ("moving backward", "i'm moving [mask]", "backward"),
        # ("moving left", "i'm moving [mask]", "left"),
        # ("moving right", "i'm moving [mask]", "right"),
        # ("jumping", "i'm [mask]", "jump"),
        # ("climbing", "i'm [mask]", "climb"),
        # ("swimming", "i'm [mask]", "swim"),
    ]
    step = 0
    for i in range(len(texts)):
        texts[i] = (
            texts[i][0],
            texts[i][1].lower().replace(",", " ").replace(".", " ").replace("?", " ").replace("[mask]", "[MASK]"),
            texts[i][2].lower().replace(",", " ").replace(".", " ").replace("?", " ").replace("[mask]", "[MASK]")
        )
    encodeds = [
        (
            tokenizer(
                text,
                add_special_tokens=True,
                return_tensors="pt",
            ),
            tokenizer.encode(answer)[1]
        ) for _, text, answer in texts
    ]


    # calculate rewards
    bias_rews = [0 for _ in range(len(texts))]
    rews = [0 for _ in range(len(texts))]
    for i in tqdm(range(len(frames))):
        if (i - sample_rate * args.n_frames) >= 0 and (i - sample_rate * args.n_frames) % (args.sample_interval * frame_rate) == 0:
            # print (i, "/", len(frames), "[", (i - sample_rate * args.n_frames) // sample_rate,  i // sample_rate, "]")

            # calculate prob bias
            if args.stack == "mean":
                video = features[:, (i - sample_rate * args.n_frames) // sample_rate: i // sample_rate]
                origin_shape = video.shape
                video = video.mean(dim=1)
                video = video.broadcast_to(origin_shape)
            elif args.stack == "first":
                video = features[:, (i - sample_rate * args.n_frames) // sample_rate: (i - sample_rate * args.n_frames) // sample_rate + 1]
                video = video.broadcast_to([video.shape[0], video.shape[1] * args.n_frames, video.shape[2]])
            video_len = torch.tensor(video.size(1), device=device)
            video_mask = get_mask(video_len, video.size(1)).to(device)
            for num, (encoded, answer_id) in enumerate(encodeds):
                input_ids = encoded["input_ids"].clone().to(device)
                while True:
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
                    
                    # print("bias", logits.softmax(dim=-1)[:, min_idx, answer_id])
                    bias_rews[num] = logits.softmax(dim=-1)[:, min_idx, answer_id].item()

                    # generate one word at a time
                    input_ids[0][min_idx] = encoded_output[0][min_idx]
                    # print(min_idx, input_ids)


            # calculate rewards
            video = features[:, (i - sample_rate * args.n_frames) // sample_rate: i // sample_rate]
            video_len = torch.tensor(video.size(1), device=device)
            video_mask = get_mask(video_len, video.size(1)).to(device)
            for num, (encoded, answer_id) in enumerate(encodeds):
                input_ids = encoded["input_ids"].clone().to(device)
                while True:
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
                    
                    # print(logits.softmax(dim=-1)[:, min_idx, answer_id])
                    rews[num] = logits.softmax(dim=-1)[:, min_idx, answer_id].item()

                    # generate one word at a time
                    input_ids[0][min_idx] = encoded_output[0][min_idx]
                    # print(min_idx, input_ids)
            step += 1
                
        for j, item in enumerate(texts):
            display, text, answer = item
            cv2.putText(frames[i], 
                "{}: {}, {:.2f}".format(step, display, (rews[j] - bias_rews[j]) * 1000), 
                (50, 20 + 30 * j), 
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
    parser.add_argument("--frames_per_second", default=5, type=int)
    parser.add_argument("--n_frames", default=15, type=int)
    parser.add_argument("--stack", default="mean", type=str)
    parser.add_argument("--answer_bias_weight", default=0, type=float)
    parser.add_argument("--sample_interval", default=1, type=int)
    args = parser.parse_args()
    args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
