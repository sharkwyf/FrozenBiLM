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
        frames.append(cv2.resize(frame, clip_param["resolution"]))
    frames = np.stack(frames)
    cap.release()

    result = cv2.VideoWriter(f"{args.output_dir}{Path(args.video_path).stem}_{Path(args.load).stem}_w{args.answer_bias_weight}.avi",
        cv2.VideoWriter_fourcc(*'MJPG'),
        frame_rate, size)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Extract features
    # L, H, W, C -> L, C, H, W
    t_frames = torch.tensor(frames.transpose(0, 3, 1, 2), dtype=torch.float, device=device)
    # L, H, W, C -> L / fps, 768
    features = clip_model.forward_image_features(t_frames[::frame_rate]).unsqueeze(0)
    print("len of features: ", features.shape, "dtype: ", features.dtype)

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

        # to be clear, exactly, for now, let's think step by step, in minecraft
        # Action
        "let's think step by step, in minecraft, exactly, for now, I am in the [MASK] biome",
        # "let's think step by step, in minecraft, exactly, for now, I just made a [MASK] [MASK] [MASK] successfully",
        # "let's think step by step, in minecraft, exactly, for now, I just made a [MASK] [MASK] [MASK] successfully",
        "let's think step by step, in minecraft, exactly, for now, I just found a [MASK] [MASK] [MASK] successfully",
        # "in minecraft. I was walking. to be more specific, I'm [MASK] [MASK] [MASK] right now.",
        # "what I'm doing in minecraft now is I'm [MASK] [MASK] [MASK] right now.",
        # "what am I doing in minecraft now? to be clear, I'm [MASK] [MASK] [MASK] right now.",
        # "to be clear, I'm [MASK] [MASK] [MASK] right now successfully.",
        # "what I was doing in minecraft is I was [MASK] [MASK] [MASK] successfully.",
        # "what was I doing in minecraft? I was [MASK] [MASK] [MASK] successfully.",
        # "in minecraft, I was [MASK] [MASK] [MASK] successfully in the past few seconds.",
        # "what I just did in minecraft is I [MASK] [MASK] [MASK] successfully.",
        # "what did I just do in minecraft? I just [MASK] [MASK] [MASK] successfully.",
        # "in minecraft, I just [MASK] [MASK] [MASK] successfully in the past few seconds.",
        # "what I have just done in minecraft is I have just [MASK] [MASK] [MASK] successfully.",
        # "what have I just done in minecraft? I have just [MASK] [MASK] [MASK] successfully.",
        # "in minecraft, I have just [MASK] [MASK] [MASK] in the past few seconds.",
        # Objects
        "what's moving? a [MASK] is moving",
        "i met a [MASK]",
        "what's the animal, for now, there is a [MASK] over there",
        "what's the animal, for now, there is a [MASK] here",
        "what's the animal, for now, there is a [MASK] afront of me",
        "what's the animal, for now, there is a [MASK] in front of me",
        "what's the animal, for now, there is a [MASK] before me",
        "for now, I'm chasing the [MASK] in front of me",
        "for now, I'm chasing the [MASK] before me",
        "for now, I'm chasing the [MASK] afront of me",
        "for now, I'm chasing the [MASK] here",
        "for now, a [MASK] is looking at me",
        "for now, a [MASK] is staring at me",
        "for now, a [MASK] is watching me",
        "for now, a [MASK] is standing before me",
        # "let's think step by step, in minecraft, exactly, for now, I see [MASK] [MASK] afront of me now, and there is [MASK] [MASK] besides that.",
        # "let's think step by step, in minecraft, exactly, for now, I have [MASK] [MASK] in my hand now.",
        # "let's think step by step, in minecraft, exactly, for now, I'm holding [MASK] [MASK] in my hand now.",
        # "let's think step by step, in minecraft, exactly, for now, I'm using [MASK] [MASK] in my hand to [MASK] [MASK] right now.",
        # "let's think step by step, in minecraft, exactly, for now, I just obtained [MASK] [MASK] [MASK] from [MASK] [MASK] in the past few seconds .",
        # Inference
        # "let's think step by step, in minecraft, in order to craft a diamond axe, for now, the next step is to [MASK] [MASK] [MASK].",
        # "let's think step by step, in minecraft, in order to find a cave, for now, the next step is to [MASK] [MASK] [MASK].",
        # "let's think step by step, in minecraft, in order to make a waterfall, for now, the next step is to [MASK] [MASK] [MASK].",
        # "let's think step by step, in minecraft, in order to build an animal pen, for now, the next step is to [MASK] [MASK] [MASK].",
        # "let's think step by step, in minecraft, in order to build a house, for now, the next step is to [MASK] [MASK] [MASK].",
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
            print("\n".join(rt))
            step += 1
                
        for j, text in enumerate(rt):
            cv2.putText(frames[i], 
                "{}: {}".format(step, text), 
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
    parser.add_argument("--answer_bias_weight", default=100, type=float)
    parser.add_argument("--sample_interval", default=5, type=int)
    args = parser.parse_args()
    args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
