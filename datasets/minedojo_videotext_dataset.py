import torch as th
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
import os
import json
from functools import partial
from torch.multiprocessing import Manager, Queue
from util.pertrel_oss_helper import init_clients
from util.misc import mask_minedojo_tokens
from util.verb_noun import ALL_NOUNS, ALL_VERBS


class Minedojo_VideoText_Dataset(Dataset):
    def __init__(self, tokenizer, features_path, *, max_feats=16, features_dim=768,
        text_start=-40, text_end=24, vid_start=-8, vid_end=8, n_process=8,
        mask_noun_prob=0.15, mask_verb_prob=0.15):
        # with open(video_index_file) as f:
        #     self.video_indices = json.load(f)
        self._tokenizer = tokenizer
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
        self._text_start = text_start
        self._text_end = text_end
        self._vid_start = vid_start
        self._vid_end = vid_end
        self._noun_ids = [ids[0] for ids in tokenizer(list(ALL_NOUNS), add_special_tokens=False)["input_ids"]]
        self._verb_ids =  [ids[0] for ids in tokenizer(list(ALL_VERBS), add_special_tokens=False)["input_ids"]]
        print("nouns:", sorted(tokenizer.batch_decode(self._noun_ids)))
        print("verbs:", sorted(tokenizer.batch_decode(self._verb_ids)))
        self._special_ids = {
            "nouns": (self._noun_ids, mask_noun_prob),
            "verbs": (self._verb_ids, mask_verb_prob),
        }
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clt_idx = self._available_clt_indices.get()
        data = self._clients[clt_idx].load_npz(f"{self._features_path}{self.data[idx]}").item()
        self._available_clt_indices.put(clt_idx)
        frames, words, starts, lens = data["feats"], data["words"], data["starts"], data["lens"]

        # process texts
        masked = (self._text_start < starts) & (starts <= self._text_end)
        # pre_masked = starts <= self._vid_start
        # in_masked = (self._vid_start < starts) & (starts <= self._vid_end)
        # post_masked = self._vid_end < starts
        # noun_masks = np.zeros(words.shape, dtype=bool)
        # verb_masks = np.zeros(words.shape, dtype=bool)
        # for noun in self._keyword_nouns:
        #     noun_masks |= words == noun
        # for verb in self._keyword_verbs:
        #     verb_masks |= words == verb

        # process videos
        try:
            video = th.from_numpy(frames).float()
            start_idx, end_idx = int((self._vid_start + 8) * 4), int((self._vid_end + 8) * 4)
            indices = sorted(np.random.choice(end_idx - start_idx - 2, self._max_feats - 2, replace=False) + start_idx + 1)
            indices = [start_idx] + indices + [end_idx]
            if len(video) > self._max_feats:
                sampled = []
                for j in indices:
                    sampled.append(video[j])
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

        return {"video": video, "video_len": video_len, "words": words[masked]}


    def minedojo_videotext_collate_fn(self, args, batch):
        bs = len(batch)
        video = th.stack([batch[i]["video"] for i in range(bs)])
        video_len = th.tensor([batch[i]["video_len"] for i in range(bs)], dtype=th.long)
        texts = [" ".join(batch[i]["words"]) for i in range(bs)]

        encoded = self._tokenizer(
            texts,
            add_special_tokens=True,
            max_length=args.max_tokens,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        inputs, labels = mask_minedojo_tokens(
            encoded["input_ids"],
            self._tokenizer,
            mlm_probability=args.minedojo_mask_probs[1], 
            special_ids=self._special_ids
        )

        return {
            "video": video,
            "video_len": video_len,
            "inputs": inputs,
            "labels": labels,
            "attention_mask": encoded["attention_mask"]
        }


def build_minedojo_videotext_dataset(args, tokenizer):
    full_dataset = Minedojo_VideoText_Dataset(
        tokenizer=tokenizer,
        features_path=args.minedojo_features_path,
        max_feats=args.max_feats,
        features_dim=args.features_dim,
        text_start=args.minedojo_text_start,
        text_end=args.minedojo_text_end,
        vid_start=args.minedojo_vid_start,
        vid_end=args.minedojo_vid_end,
        mask_noun_prob=args.word_mask_probs[0],
        mask_verb_prob=args.word_mask_probs[1],
    )
    train_size = int(len(full_dataset) * 0.9)
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=th.Generator().manual_seed(42))
    return train_dataset, test_dataset, partial(full_dataset.minedojo_videotext_collate_fn, args)
