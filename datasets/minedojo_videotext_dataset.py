import torch as th
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
import os


class Minedojo_VideoText_Dataset(Dataset):
    def __init__(self, features_path, max_feats=10, features_dim=768, start=-40, end=24, vid_start=-8, vid_end=8):
        features = np.load(features_path, allow_pickle=True).item()
        self.keywords = list(features.keys())
        self.data = []
        for keyword in self.keywords:
            self.data.extend(list(features[keyword].values()))
        self.max_feats = max_feats
        self.features_dim = features_dim
        self.start = start
        self.end = end
        self.vid_start = -8
        self.vid_end = 8

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames, captions = self.data[idx]
        text = captions["word"]
        masked = (self.start < captions["start"]) & (captions["start"] <= self.end)
        pre_masked = captions["start"] <= self.vid_start
        in_masked = (self.vid_start < captions["start"]) & (captions["start"] <= self.vid_end)
        post_masked = self.vid_end < captions["start"]

        pre_text = " ".join(text[masked & pre_masked])
        in_text = " ".join(text[masked & in_masked])
        post_text = " ".join(text[masked & post_masked])

        try:
            video = th.from_numpy(frames).float()
            indices = sorted(np.random.choice(video.shape[0] - 2, self.max_feats - 2, replace=False) + 1)
            if len(video) > self.max_feats:
                sampled = [video[0]]
                for j in indices:
                    sampled.append(video[j])
                sampled += [video[-1]]
                video = th.stack(sampled)
                video_len = self.max_feats
            elif len(video) < self.max_feats:
                video_len = len(video)
                video = th.cat(
                    [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
                )
            else:
                video_len = self.max_feats
        except:  # missing video or corrupted feature file
            video = th.zeros(self.max_feats, self.features_dim)
            video_len = 0

        return {"video": video, "video_len": video_len, "pre_text": pre_text, "in_text": in_text, "post_text": post_text}


def minedojo_videotext_collate_fn(batch):
    bs = len(batch)
    video = th.stack([batch[i]["video"] for i in range(bs)])
    video_len = th.tensor([batch[i]["video_len"] for i in range(bs)], dtype=th.long)
    pre_text = [batch[i]["pre_text"] for i in range(bs)]
    in_text = [batch[i]["in_text"] for i in range(bs)]
    post_text = [batch[i]["post_text"] for i in range(bs)]

    return {
        "video": video,
        "video_len": video_len,
        "pre_text": pre_text,
        "in_text": in_text,
        "post_text": post_text,
    }


def build_minedojo_videotext_dataset(args):
    full_dataset = Minedojo_VideoText_Dataset(
        features_path=args.minedojo_features_path,
        max_feats=args.max_feats,
        features_dim=args.features_dim,
        start=args.minedojo_text_start,
        end=args.minedojo_text_end,
    )
    train_size = int(len(full_dataset) * 0.9)
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=th.Generator().manual_seed(42))
    return train_dataset, test_dataset
