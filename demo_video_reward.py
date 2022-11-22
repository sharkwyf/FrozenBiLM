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
from util.verb_noun import ALL_NOUNS, ALL_VERBS
from benchmark_eval import benchmark_evaluate

questions = {
    "nouns": [
        ("i see the [MASK]", ""),
        ("i find the [MASK]", ""),
        ("i'm watching the [MASK]", ""),
        ("i'm looking at the [MASK]", ""),
        ("the [MASK] is before me", ""),
        ("the [MASK] is in front of me", ""),
    ],
    "verbs": [
        ("i am [MASK]", "present"),
        ("i am just [MASK]", "present"),
        ("i was [MASK]", "present"),
        ("i was just [MASK]", "present"),
        ("what i'm doing is i'm just [MASK]", "present"),
        ("what i was doing is i was just [MASK]", "present"),
    ],
}

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
    clip_model.eval()

    # load frozenbilm model
    model = build_model(args)
    model.to(device)
    model.eval()
    tokenizer = get_tokenizer(args)

    # encoded available words
    # encoded available words
    answer_bias_dict = {}
    for type, words in [("nouns", ALL_NOUNS), ("verbs", ALL_VERBS)]:
        answer_id = tokenizer.encode(["▁" + w for w in list(words)])[1:-1]
        answer_bias = torch.zeros(tokenizer.vocab_size, dtype=torch.float32, device=device)
        answer_bias += args.answer_bias_weight
        for id in answer_id:
            answer_bias[id] = 0
        answer_bias_dict[type] = answer_bias

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
        "{}{}_{}_ckpt{}_{}_t{}_b{}_reward.avi".format(
            args.output_dir,
            Path(args.video_path).stem,
            "_".join(args.load.split('/')[-3:-1]),
            Path(args.load).stem[-2:],
            args.stack,
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
    
    step = 0
    # calculate rewards
    for i in tqdm(range(len(frames))):
        if (i - sample_rate * args.n_frames // 2) >= 0 and (i - sample_rate * args.n_frames // 2) % (args.sample_interval * frame_rate) == 0:
            print(i, "/", len(frames), "[", (i - sample_rate * args.n_frames // 2) // sample_rate,  (i + sample_rate * args.n_frames // 2) // sample_rate, "]")
            video = features[:, (i - sample_rate * args.n_frames // 2) // sample_rate: (i + sample_rate * args.n_frames // 2) // sample_rate]\

            data = {
                "verbs": {
                    "run": [video],
                    "stand": [video],
                    "jump": [video],
                    "climb": [video],
                    "attack": [video],
                    "mine": [video],
                    "chop": [video],
                    "dig": [video],
                    "watch": [video],
                }
            }

            benchmark_stats, words_stats = benchmark_evaluate(
                model=model,
                tokenizer=tokenizer,
                data=data,
                answer_bias_dict=answer_bias_dict,
                args=args,
                questions=questions,
                device=device,
            )
            
            step += 1
            
        if (i - sample_rate * args.n_frames // 2) >= 0:   
            for j, item in enumerate(words_stats["verbs"].items()):
                display, info = item
                cv2.putText(frames[i], 
                    "{}: {}, {:.4f}".format(step, display, info["conf"] * 100), 
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
    parser.add_argument("--frames_per_second", default=2, type=int)
    parser.add_argument("--n_frames", default=16, type=int)
    parser.add_argument("--stack", default=["none", "first", "mean"][0], type=str)
    parser.add_argument("--answer_bias_weight", default=-100, type=float)
    parser.add_argument("--sample_interval", default=1, type=int)
    args = parser.parse_args()
    args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
