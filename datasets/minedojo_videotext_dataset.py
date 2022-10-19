import torch as th
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
import os


class Minedojo_VideoText_Dataset(Dataset):
    def __init__(self, features_path, max_feats=10, features_dim=768):
        features = np.load(features_path, allow_pickle=True).item()
        self.keywords = list(features.keys())
        self.data = []
        for keyword in self.keywords:
            self.data.extend(list(features[keyword].values()))
        self.max_feats = max_feats
        self.features_dim = features_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames, captions = self.data[idx]
        text = captions["word"]
        text = " ".join(text)

        try:
            video = th.from_numpy(frames).float()
            if len(video) > self.max_feats:
                sampled = []
                for j in range(self.max_feats):
                    sampled.append(video[(j * len(video)) // self.max_feats])
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

        return {"video": video, "video_len": video_len, "text": text}


def minedojo_videotext_collate_fn(batch):
    bs = len(batch)
    video = th.stack([batch[i]["video"] for i in range(bs)])
    video_len = th.tensor([batch[i]["video_len"] for i in range(bs)], dtype=th.long)
    text = [batch[i]["text"] for i in range(bs)]

    return {
        "video": video,
        "video_len": video_len,
        "text": text,
    }


def build_minedojo_videotext_dataset(args):
    full_dataset = Minedojo_VideoText_Dataset(
        features_path=args.minedojo_features_path,
        max_feats=args.max_feats,
        features_dim=args.features_dim,
    )
    train_size = int(len(full_dataset) * 0.9)
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=th.Generator().manual_seed(42))
    return train_dataset, test_dataset
