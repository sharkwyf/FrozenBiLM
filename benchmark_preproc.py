import os
import torch
import torch.nn
import torch.optim
import numpy as np
import cv2
import argparse
import glob
from pathlib import Path

from tqdm import tqdm
from model.mineclip import MineCLIP, utils as U


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
    
    types = ["nouns", "verbs"]
    result = { type: {} for type in types}
    for type in types:
        print("processing", type)
        for file in tqdm(glob.glob(args.video_dir + type + "/*.mp4")):
            # print(file)
            
            # Extract frames from video
            cap = cv2.VideoCapture(file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            size = (frame_width, frame_height)

            resized = [256, 160]
            resized_frames = []
            indices = np.linspace(0, total_frames - 2, num=args.n_frames, dtype=int)
            for idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if idx in indices:
                    resized_frames.append(cv2.resize(frame, resized))
            resized_frames = np.stack(resized_frames)
            cap.release()

            # Extract features
            # L, H, W, C -> L, C, H, W
            t_frames = U.any_to_torch_tensor(resized_frames.transpose(0, 3, 1, 2), dtype=torch.float, device=device)
            features = clip_model.forward_image_features(t_frames).cpu().numpy()
            features = torch.from_numpy(features).float().unsqueeze(0)
            assert features.shape[1] == 16, f"len of features: {features.shape} dtype: {features.dtype}"

            label = Path(file).stem[:-1]
            if label not in result[type]:
                result[type][label] = [features]
            else:
                result[type][label].extend([features])

    np.save(args.output_path, result)

    # close all windows
    cv2.destroyAllWindows()
    print("Succeed")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="./workspace/FrozenBiLM/data/Minedojo/benchmarks/", type=str)
    parser.add_argument("--output_path", default="./workspace/FrozenBiLM/data/Minedojo/benchmarks/features.npy", type=str)
    parser.add_argument("--model_path", default="./workspace/FrozenBiLM/data/Minedojo/attn.pth", type=str)
    # parser.add_argument("--frames_per_second", default=4, type=int)
    parser.add_argument("--n_frames", default=16, type=int)
    args = parser.parse_args()
    main(args)
