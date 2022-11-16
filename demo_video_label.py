import os
import torch
import torch.nn
import torch.optim
import numpy as np
import cv2
import argparse
from pathlib import Path

from tqdm import tqdm
from model import build_model, get_tokenizer
from args import get_args_parser
from model.mineclip import MineCLIP, utils as U
from util.misc import get_mask
from util.verb_noun import ALL_WORDS


"""
Label intentions given a video 
"""
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

    result = cv2.VideoWriter(
        "{}{}_{}_ckpt{}_t{}_b{}_label.avi".format(
            args.output_dir,
            Path(args.video_path).stem,
            "_".join(args.load.split('/')[-3:-1]),
            Path(args.load).stem[-2:],
            args.n_frames // args.frames_per_second,
            args.answer_bias_weight
        ),
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

    # to be clear, exactly, for now, let's think step by step, in minecraft
    texts = [

        # Animals
        # "I found a [mask] [mask] [mask]",
        # "I found a [mask] [mask] now",
        # "what's the animal, for now, there is a [MASK] over there",
        # "what's the animal, for now, there is a [MASK] afront of me",
        # "what's the animal, for now, there is a [MASK] in front of me",
        # "what's the animal, for now, there is a [MASK] before me",
        # "for now, a [MASK] is looking at me",
        # "for now, a [MASK] is staring at me",

        # Waterfall
        "I am in the [MASK] biome",
        "I just made a [MASK] [MASK] [MASK] successfully",
        "I just found a [MASK] [MASK] [MASK] successfully",
        "in minecraft. I'm [MASK] [MASK] [MASK] right now.",
        "what I'm doing in minecraft now is I'm [MASK] [MASK] [MASK] right now.",
        "what am I doing in minecraft now? to be clear, I'm [MASK] [MASK] [MASK] right now.",
        "to be clear, I'm [MASK] [MASK] [MASK] right now successfully.",
        "what I was doing in minecraft is I was [MASK] [MASK] [MASK] successfully.",
        "what was I doing in minecraft? I was [MASK] [MASK] [MASK] successfully.",
        "in minecraft, I was [MASK] [MASK] [MASK] successfully in the past few seconds.",
        "what I just did in minecraft is I [MASK] [MASK] [MASK] successfully.",
        "what did I just do in minecraft? I just [MASK] [MASK] [MASK] successfully.",
        "in minecraft, I just [MASK] [MASK] [MASK] successfully in the past few seconds.",
        "what I have just done in minecraft is I have just [MASK] [MASK] [MASK] successfully.",
        "in minecraft, I have just [MASK] [MASK] [MASK] in the past few seconds.",
        # Objects
        
        # Common
        "I see [MASK] [MASK] afront of me now, and there is [MASK] [MASK] besides that.",
        "I have [MASK] [MASK] in my hand now.",
        "I'm holding [MASK] [MASK] in my hand now.",
        "I had [MASK] [MASK] [MASK] in my hand",
        "I have [MASK] [MASK] [MASK] in my hand",
        "I just got [MASK] [MASK] [MASK]",
        "I just obtained [MASK] [MASK] [MASK]",
        "In minecraft, I just [MASK] [MASK] [MASK]",
    ]
    step = 0
    texts = [t.lower().replace(",", " ").replace(".", " ").replace("?", " ").replace("[mask]", "[MASK]") for t in texts]
    rt = [t for t in texts]
    for i in tqdm(range(len(frames))):
        if (i - sample_rate * args.n_frames) >= 0 and (i - sample_rate * args.n_frames) % (args.sample_interval * frame_rate) == 0:
            print (i, "/", len(frames), "[", (i - sample_rate * args.n_frames) // sample_rate,  i // sample_rate, "]")
            video = features[:, (i - sample_rate * args.n_frames) // sample_rate: i // sample_rate]
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

                    # generate one word at a time
                    input_ids[0][min_idx] = encoded_output[0][min_idx]
                    # print(min_idx, input_ids)
                rt[num] = tokenizer.batch_decode(input_ids[:, 1:-1])[0]
            print("\n".join(rt))
            step += 1
             
        if i - (args.n_frames // 2) >= 0:   
            for j, text in enumerate(rt):
                cv2.putText(frames[i - (args.n_frames // 2)], 
                    "{}: {}".format(step, text), 
                    (50, 20 + 30 * j), 
                    font, 0.6, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_4)
            result.write(frames[i - (args.n_frames // 2)])

    for i in range(i - (args.n_frames // 2), len(frames)):
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
    parser.add_argument("--frames_per_second", default=4, type=int)
    parser.add_argument("--n_frames", default=16, type=int)
    parser.add_argument("--answer_bias_weight", default=0, type=float)
    parser.add_argument("--sample_interval", default=4, type=int)
    args = parser.parse_args()
    args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
