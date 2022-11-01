import torch as th
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
import os
import json
from torch.multiprocessing import Manager, Queue
from util.pertrel_oss_helper import init_clients


class Minedojo_VideoText_Dataset(Dataset):
    def __init__(self, video_index_file, features_path, *, max_feats=16, features_dim=768, start=-40, end=24, vid_start=-8, vid_end=8, n_process=8):
        with open(video_index_file) as f:
            self.video_indices = json.load(f)
        self._clients = init_clients(n_process)
        self._available_clt_indices = Queue(len(self._clients))
        for i in range(len(self._clients)):
            self._available_clt_indices.put(i)
        
        print(f"fetching data indices from {features_path}")
        self.data = list(self._clients[0].list(features_path))
        print(f"fetched {len(self.data)} indices")

        self._features_path = features_path
        self._max_feats = max_feats
        self._features_dim = features_dim
        self._start = start
        self._end = end
        self._vid_start = vid_start
        self._vid_end = vid_end
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clt_idx = self._available_clt_indices.get()
        data = self._clients[clt_idx].load_npz(f"{self._features_path}{self.data[idx]}").item()
        self._available_clt_indices.put(clt_idx)
        frames, words, starts, lens = data["feats"], data["words"], data["starts"], data["lens"]

        masked = (self._start < starts) & (starts <= self._end)
        pre_masked = starts <= self._vid_start
        in_masked = (self._vid_start < starts) & (starts <= self._vid_end)
        post_masked = self._vid_end < starts

        pre_text = " ".join(words[masked & pre_masked])
        in_text = " ".join(words[masked & in_masked])
        post_text = " ".join(words[masked & post_masked])

        try:
            video = th.from_numpy(frames).float()
            indices = sorted(np.random.choice(video.shape[0] - 2, self._max_feats - 2, replace=False) + 1)
            if len(video) > self._max_feats:
                sampled = [video[0]]
                for j in indices:
                    sampled.append(video[j])
                sampled += [video[-1]]
                video = th.stack(sampled)
                video_len = self._max_feats
            elif len(video) < self._max_feats:
                video_len = len(video)
                video = th.cat(
                    [video, th.zeros(self._max_feats - video_len, self._features_dim)], 0
                )
            else:
                video_len = self._max_feats
        except:  # missing video or corrupted feature file
            video = th.zeros(self._max_feats, self._features_dim)
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
        video_index_file=args.video_index_file,
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
