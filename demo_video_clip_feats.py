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
        # BGR -> RGB
        frame = frame[..., ::-1]
        frames.append(frame)
        resized_frames.append(cv2.resize(frame, resized))
    frames = np.stack(frames)
    resized_frames = np.stack(resized_frames)
    cap.release()

    # calculate feats
    t_frames = U.any_to_torch_tensor(resized_frames.transpose(0, 3, 1, 2), dtype=torch.float, device=device)
    # L, H, W, C -> L, 768
    features = clip_model.forward_image_features(t_frames).cpu().numpy()
    features = torch.from_numpy(features).float()
    print(features.shape)
    span = (-2, 2)
    diffs = features[:-1] - features[1:]
    # diffs1 = diffs[:-1] - diffs[1:]
    diffs = torch.linalg.norm(diffs, dim=1)
    diffs = (diffs[:-1] - diffs[1:]).abs()

    result = cv2.VideoWriter(
        "{}{}_feats.avi".format(
            args.output_dir,
            Path(args.video_path).stem,
        ),
        cv2.VideoWriter_fourcc(*'MJPG'),
        frame_rate, size)

    for i in tqdm(range(1, len(frames))):
        cv2.putText(frames[i], 
            "{}: {:.5f}".format(i, diffs[max(i - 1 + span[0], 0) : min(i + span[1], len(diffs) - 1)].mean()), 
            (50, 20 + 30 * 0), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="./workspace/FrozenBiLM/data/Minedojo/verbs.mp4", type=str)
    parser.add_argument("--output_dir", default="./workspace/FrozenBiLM/data/Minedojo/output/", type=str)
    parser.add_argument("--model_path", default="./workspace/FrozenBiLM/data/Minedojo/attn.pth", type=str)
    args = parser.parse_args()
    main(args)
