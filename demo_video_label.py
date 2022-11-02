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
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    frames = np.stack(frames)
    cap.release()

    result = cv2.VideoWriter(f"{args.output_dir}{Path(args.video_path).stem}_{Path(args.load).stem}_w{args.answer_bias_weight}.avi",
        cv2.VideoWriter_fourcc(*'MJPG'),
        frame_rate, size)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Extract features
    # L, H, W, C -> L, C, H, W
    t_frames = torch.tensor(frames.transpose(0, 3, 1, 2), dtype=float, device=device)
    t_frames = resize_frames(t_frames, clip_param["resolution"])
    # L, H, W, C -> L / fps, 768
    features = clip_model.forward_image_features(t_frames[::frame_rate]).unsqueeze(0)
    print("len of features: ", features.shape)

    texts = [
        # "I saw [MASK] [MASK] [MASK] afront of me",
        # "I had [MASK] [MASK] [MASK] in my hand",
        # "I have [MASK] [MASK] [MASK] in my hand",
        # "I just got [MASK] [MASK] [MASK]",
        # "I just obtained [MASK] [MASK] [MASK]",
        # "What was I doing? I was [MASK] [MASK] [MASK]",
        # "In minecraft, I just [MASK] [MASK] [MASK]",
        # "Let's think step by step. In minecraft, now I should [MASK] [MASK] [MASK] to craft a diamond axe",
        # "Let's think step by step. In minecraft, now I should [MASK] [MASK] [MASK] to find a cave",
        # "Let's think step by step. In minecraft, now I should [MASK] [MASK] [MASK] to make a waterfall",
        # "Let's think step by step. In minecraft, now I should [MASK] [MASK] [MASK] to build an animal pen",
        # "Let's think step by step. In minecraft, now I should [MASK] [MASK] [MASK] to build a house",

        # Action
        "What I am doing is I am [MASK] [MASK] [MASK] .",
        "What am I doing? I am [MASK] [MASK] [MASK] .",
        "I am [MASK] [MASK] [MASK] just now."
        "What I was doing is I [MASK] [MASK] [MASK] .",
        "What was I doing? I [MASK] [MASK] [MASK] .",
        "I was [MASK] [MASK] [MASK] in past few seconds."
        "What I just did is I [MASK] [MASK] [MASK] .",
        "What did I just do? I [MASK] [MASK] [MASK] .",
        "I just [MASK] [MASK] [MASK] in past few seconds."
        "What I have just done is I [MASK] [MASK] [MASK] .",
        "What have I just done? I [MASK] [MASK] [MASK] .",
        "I have just [MASK] [MASK] [MASK] in past few seconds."
        # Objects
        "I am using [MASK] [MASK] [MASK] .",
        "I saw [MASK] [MASK] [MASK] afront of me, and [MASK] [MASK] [MASK] besides .",
        "I have [MASK] [MASK] [MASK] in my hand .",
        "I just obtained [MASK] [MASK] [MASK] in past few seconds.",
        # Inference
        "Let's think step by step. In minecraft, now I should [MASK] [MASK] [MASK] to craft a diamond axe .",
        "Let's think step by step. In minecraft, now I should [MASK] [MASK] [MASK] to find a cave .",
        "Let's think step by step. In minecraft, now I should [MASK] [MASK] [MASK] to make a waterfall .",
        "Let's think step by step. In minecraft, now I should [MASK] [MASK] [MASK] to build an animal pen .",
        "Let's think step by step. In minecraft, now I should [MASK] [MASK] [MASK] to build a house .",
    ]
    step = 0
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
                    input_ids[0][min_idx] = encoded_output[0][min_idx]
                    # print(min_idx, input_ids)
                rt[num] = tokenizer.batch_decode(input_ids[:, 1:-1])[0]
            print(rt)
            step += 1
                
        for j, text in enumerate(rt):
            cv2.putText(frames[i], 
                "{}: {}".format(step, text), 
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
    parser.add_argument("--answer_bias_weight", default=100, type=float)
    parser.add_argument("--sample_interval", default=5, type=int)
    parser.add_argument("--half_precision", type=bool, default=True, help="whether to output half precision float or not")
    args = parser.parse_args()
    args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
