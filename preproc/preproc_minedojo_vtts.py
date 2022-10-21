import os
import webvtt
import re
import random
import argparse
import multiprocessing as mp
from multiprocessing import Process, Lock, Queue, Pool
from collections import deque, Counter
# import imageio
import json
import time
from io import StringIO, BytesIO
from petrel_client.client import Client
from tqdm import tqdm
import numpy as np


"""
Load VTT files and annotate clip ranges
"""
def process_keywords(path):
    with open(path) as json_file:
        return json.load(json_file)

def process(captions):
    ret = []
    for ca in captions:
        ca.text = ca.text.split('\n')[-1]
        if ca.text.strip():
            ret.append(ca)
    return ret

def preproc_minedojo_vtts(input):
    rank = mp.current_process()._identity[0] - 1
    args, includes, excludes, keyid = input

    payload = clients[rank].get(os.path.join(args.input_path, f"{keyid}{args.vtt_suffix}"))
    try:
        payload = payload.decode("utf-8")
    except Exception as e:
        with open('error.log', 'a') as f1:
            f1.write(keyid+'\n')
        return 0, None
    vtt = webvtt.read_buffer(StringIO(payload))
    words, words_start, words_lens = [], [], []
    for sentence in process(vtt.captions):
        t_start, t_len = sentence.start_in_seconds, sentence.end_in_seconds - sentence.start_in_seconds
        _words = np.array(sentence.text.split())
        if (t_len / len(_words)) < args.min_word_interval:
            continue
        _word_starts = np.arange(len(_words)) * t_len / len(_words) + t_start
        _word_lens = np.ones(len(_words)) * t_len / len(_words)
        words.append(_words)
        words_start.append(_word_starts)
        words_lens.append(_word_lens)
    if len(words) == 0:
        return 0, None
    words = np.concatenate(words)
    words_start = np.concatenate(words_start)
    words_lens = np.concatenate(words_lens)

    match1 = np.zeros(len(words), dtype=bool)
    match2 = np.zeros(len(words) - 1, dtype=bool)
    dismatch2 = np.zeros(len(words) - 1, dtype=bool)
    words2 = np.stack([words[:-1], words[1:]]).transpose()
    for include in includes[1]:
        match1 += words == include
    for include in includes[2]:
        match2 += (words2 == include).all(axis=1)
    for exclude in excludes[2]:
        dismatch2 += (words2 == exclude).all(axis=1)
    dismatch2 = ~dismatch2
    match1[:-1] &= dismatch2
    match1[1:] &= dismatch2
    match1[:-1] += match2
    if len(words) == 0:
        return 0, None
    rt = {}
    count = 0
    prev = -args.min_clip_interval
    for idx in match1.nonzero()[0]:
        keyword, t_word_start = words[idx], words_start[idx]
        if prev + args.min_clip_interval > t_word_start:
            continue
        if keyword not in rt:
            rt[keyword] = [[keyid, round(t_word_start, 2), round(words_lens[idx], 2)]]
        else:
            rt[keyword].append([keyid, round(t_word_start, 2), round(words_lens[idx], 2)])
        count += 1
        prev = t_word_start
    return count, rt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_path', type=str, default='s3://minedojo/videos_downloading/2022092202/')
    parser.add_argument('--output_path', type=str, default='./minedojo_clips.json')
    parser.add_argument('--keyword_path', type=str, default='./keywords.json')
    parser.add_argument('--clip_start', type=float, default=-8)
    parser.add_argument('--clip_end', type=float, default=8)
    parser.add_argument('--vid_suffix', type=str, default=".mp4")
    parser.add_argument('--vtt_suffix', type=str, default=".en.vtt")
    parser.add_argument('--min_word_interval', type=float, default=0.1)
    parser.add_argument('--min_clip_interval', type=float, default=16)
    parser.add_argument('--n_process', type=int, default=2)
    args = parser.parse_args()
    print(args)
    print(f"total cpu counts: {mp.cpu_count()}")

    suffix_len = len(args.vtt_suffix)
    keywords_dict = process_keywords(args.keyword_path)
    includes = {i: [] for i in range(1, 3)}
    excludes = {i: [] for i in range(2, 3)}
    for key, items in keywords_dict.items():
        include, exclude = items["include"], items["exclude"]
        for phrase in include:
            splits = phrase.split()
            includes[len(splits)].append(np.array(splits))
        for phrase in exclude:
            splits = phrase.split()
            excludes[len(splits)].append(np.array(splits))


    clients = [Client() for _ in range(args.n_process)]
    contents = []
    print("fetching indices")
    files = set(clients[0].list(args.input_path))
    print(f"loaded {len(files)} files")
    for key in files:
        if key.endswith(args.vtt_suffix) and f"{key[:-suffix_len]}{args.vid_suffix}" in files:
            contents.append((args, includes, excludes, key[:-suffix_len]))
    
    print("starting processes")
    with Pool(processes=args.n_process) as pool:
        results = tqdm(
            pool.imap_unordered(
                preproc_minedojo_vtts,
                contents,
            ),
            total=len(contents)
        )
    
        rt = {}
        total_cnt = 0
        for _, (cnt, result) in enumerate(results):
            if cnt > 0:
                for k, v in result.items():
                    if k in rt:
                        rt[k].extend(v)
                    else:
                        rt[k] = v
                total_cnt += cnt

        with open(args.output_path, 'w') as f:
            json.dump(rt, f)
    
    counter = Counter()
    for k, v in rt.items():
        counter[k] = len(v)
    print(counter)
    print(f"total processed video clips: {total_cnt}, average clips per video: {total_cnt / len(contents)}")

