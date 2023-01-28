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
    def __init__(self, tokenizer, features_path, idm_features_path, *, max_feats=16, features_dim=768, use_idm_features=False, idm_features_dim=4096,
        text_min_range=None, text_max_range=[], vid_min_range=None, vid_max_range=[], 
        n_process=8, mask_noun_prob=0.15, mask_verb_prob=0.15):
        # with open(video_index_file) as f:
        #     self.video_indices = json.load(f)
        self._tokenizer = tokenizer
        self._clients = init_clients(n_process)
        self._available_clt_indices = Queue(len(self._clients))
        self._use_idm_features = use_idm_features
        for i in range(len(self._clients)):
            self._available_clt_indices.put(i)
        
        print(f"fetching feature indices from {features_path}")
        self.data = sorted(list(self._clients[0].list(features_path)))
        print(f"fetched {len(self.data)} indices")
        
        if self._use_idm_features:
            # load idm features
            print(f"fetching idm feature indices from {idm_features_path}")
            self.idm_data = sorted(list(self._clients[0].list(idm_features_path)))
            print(f"fetched {len(self.idm_data)} indices")

            # filter data with idm features
            print(f"filtering, with idm features")
            idms_set = set([x[:-4] for x in self.idm_data])
            self.data = sorted([x for x in self.data if x[:-4] in idms_set])
            print(f"got {len(self.data)} indices")

        self._features_path = features_path
        self._idm_features_path = idm_features_path
        self._max_feats = max_feats
        self._features_dim = features_dim
        self._idm_features_dim = idm_features_dim
        self._output_features_dim = self._features_dim + self._use_idm_features * self._idm_features_dim
        self._text_min_range = text_min_range
        self._text_max_range = text_max_range
        self._vid_min_range = vid_min_range
        self._vid_max_range = vid_max_range
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
        clip_id = self.data[idx][:-4]
        data = self._clients[clt_idx].load_npz(f"{self._features_path}{clip_id}.npz").item()
        if self._use_idm_features:
            idm = self._clients[clt_idx].load_nbz(f"{self._idm_features_path}{clip_id}.nbz")
        self._available_clt_indices.put(clt_idx)
        frames, words, starts, lens = data["feats"], data["words"], data["starts"], data["lens"]

        sample_text_start = np.random.uniform(*self._text_min_range)
        sample_text_end = np.random.uniform(*self._text_max_range)
        sample_vid_start = np.random.uniform(*self._vid_min_range)
        sample_vid_end = np.random.uniform(*self._vid_max_range)

        # process texts
        masked = (sample_text_start < starts) & (starts <= sample_text_end)
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
            if self._use_idm_features:
                idm_feats = th.from_numpy(idm).float()
                assert len(video) == len(idm_feats)
            start_idx, end_idx = int((sample_vid_start + 8) * 4), int((sample_vid_end + 8) * 4)
            if len(video) > self._max_feats:
                indices = sorted(np.random.choice(end_idx - start_idx - 2, self._max_feats - 2, replace=False) + start_idx + 1)
                indices = [start_idx] + indices + [end_idx]
                video = video[indices]
                video_len = self._max_feats
                if self._use_idm_features:
                    idm_feats = idm_feats[indices]
            elif len(video) < self._max_feats:
                video_len = len(video)
                video = th.cat(
                    [video, th.zeros(self._max_feats - video_len, self._features_dim)], 0
                )
                if self._use_idm_features:
                    idm_feats = th.cat(
                        [idm_feats, th.zeros(self._max_feats - video_len, self._idm_features_dim)], 0
                    )
            else:
                video_len = self._max_feats
        except:  # missing video or corrupted feature file
            video = th.zeros(self._max_feats, self._features_dim)
            video_len = 0
            idm_feats = th.zeros(self._max_feats, self._idm_features_dim)

        return {
            "video": video,
            "video_len": video_len,
            "idm_feats": idm_feats if self._use_idm_features else None,
            "words": words[masked]
        }


    def minedojo_videotext_collate_fn(self, args, batch):
        bs = len(batch)
        video = th.stack([batch[i]["video"] for i in range(bs)])
        video_len = th.tensor([batch[i]["video_len"] for i in range(bs)], dtype=th.long)
        idm_feats = th.stack([batch[i]["idm_feats"] for i in range(bs)])
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
            "idm_feats": idm_feats,
            "inputs": inputs,
            "labels": labels,
            "attention_mask": encoded["attention_mask"]
        }


def build_minedojo_videotext_dataset(args, tokenizer):
    full_dataset = Minedojo_VideoText_Dataset(
        tokenizer=tokenizer,
        features_path=args.minedojo_features_path,
        idm_features_path=args.minedojo_idm_features_path,
        use_idm_features=args.use_idm_features,
        max_feats=args.max_feats,
        features_dim=args.features_dim,
        text_min_range=args.minedojo_text_min_range,
        text_max_range=args.minedojo_text_max_range,
        vid_min_range=args.minedojo_vid_min_range,
        vid_max_range=args.minedojo_vid_max_range,
        mask_noun_prob=args.word_mask_probs[0],
        mask_verb_prob=args.word_mask_probs[1],
    )
    train_size = int(len(full_dataset) * 0.9)
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=th.Generator().manual_seed(42))
    return train_dataset, test_dataset, partial(full_dataset.minedojo_videotext_collate_fn, args)
