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
import sys
workdir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, workdir)
from util.pertrel_oss_helper import init_clients
from util.verb_noun import KW_VERBS


"""
Load VTT files and annotate clip ranges
"""
def load_keywords(path):
    with open(path) as json_file:
        data = json.load(json_file)
        data.update(KW_VERBS)
        return data

def process(captions):
    ret = []
    for ca in captions:
        ca.text = ca.text.split('\n')[-1]
        if ca.text.strip():
            ret.append(ca)
    return ret

def preproc_minedojo_vtts(input):
    """Extract keywords from vtt files"""
    client = clients[mp.current_process()._identity[0] - 1]
    args, includes, excludes, keyid = input

    vtt = client.load_vtt(os.path.join(args.input_path, f"{keyid}{args.vtt_suffix}"))
    if vtt is None:
        return 0, None
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

    # keyward matching
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
    match2 = np.concatenate((match2, np.zeros(1, dtype=bool)))
    dismatch2 = ~dismatch2
    match1[:-1] &= dismatch2
    match1[1:] &= dismatch2
    match1 += match2
    if len(words) == 0:
        return 0, None

    # return matched keywords
    rt = {}
    count = 0
    t_prev_start = -args.min_clip_interval
    for idx in match1.nonzero()[0]:
        keyword, t_word_start = "_".join(words[idx:idx+1+match2[idx]]), words_start[idx]
        if t_prev_start + args.min_clip_interval > t_word_start:
            if keyword not in rt:
                rt[keyword] = {(keyid, round(t_prev_start, 2)) : round(words_lens[idx], 2)}
            else:
                rt[keyword][(keyid, round(t_prev_start, 2))] = round(words_lens[idx], 2)
        else:
            if keyword not in rt:
                rt[keyword] = {(keyid, round(t_word_start, 2)) : round(words_lens[idx], 2)}
            else:
                rt[keyword][(keyid, round(t_word_start, 2))] = round(words_lens[idx], 2)
            count += 1
            t_prev_start = t_word_start
    return count, rt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_path', type=str, default='s3://minedojo/videos_test/')
    parser.add_argument('--output_path', type=str, default='./minedojo_clips.json')
    parser.add_argument('--keyword_path', type=str, default='./keywords.json')
    parser.add_argument('--vid_suffix', type=str, default=".mp4")
    parser.add_argument('--vtt_suffix', type=str, default=".en.vtt")
    parser.add_argument('--min_word_interval', type=float, default=0.1)
    parser.add_argument('--min_clip_interval', type=float, default=16)
    parser.add_argument('--n_process', type=int, default=mp.cpu_count())
    args = parser.parse_args()
    print(args)
    print(f"total cpu counts: {mp.cpu_count()}")

    # load from keywords.json and group into <include> and <exclude>
    suffix_len = len(args.vtt_suffix)
    keywords_dict = load_keywords(args.keyword_path)
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

    # fetch all filenamnes and filter available files
    clients = init_clients(args.n_process)
    contents = []
    print("fetching indices")
    files = set(clients[0].list(args.input_path))
    print(f"loaded {len(files)} files")
    for key in files:
        if key.endswith(args.vtt_suffix) and f"{key[:-suffix_len]}{args.vid_suffix}" in files:
            contents.append((args, includes, excludes, key[:-suffix_len]))
    
    # preprocess vtt files
    print("starting processes")
    with Pool(processes=args.n_process) as pool:
        results = tqdm(
            pool.imap_unordered(
                preproc_minedojo_vtts,
                contents,
            ),
            total=len(contents)
        )

        # gather processed clips and save
        rt, total_cnt = {}, 0
        for _, (cnt, result) in enumerate(results):
            if cnt > 0:
                for k, v in result.items():
                    if k not in rt:
                        rt[k] = []
                    for (keyid, t_start), t_len in v.items():
                        rt[k].append([keyid, t_start, t_len])
                total_cnt += cnt
        with open(args.output_path, 'w') as f:
            json.dump(rt, f)
    
    # print stastics
    counter = Counter()
    for k, v in rt.items():
        counter[k] = len(v)
    print(counter)
    print(f"total processed videos: {len(contents)}, total processed video clips: {total_cnt}, average clips per video: {total_cnt / len(contents)}")

