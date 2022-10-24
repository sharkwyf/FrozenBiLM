import sys
import torch
import json
import os
import cv2
import numpy as np
import webvtt
from io import StringIO
import bisect
import io
import copy
import time
import math
import argparse
import multiprocessing
from multiprocessing import Process, Lock, Queue, Pool
from tqdm import *
from petrel_client.client import Client
from pathlib import Path


"""
Load MP4 videos and extract 64 frames & captions in 64s
"""
client = Client()
debug = False

class File:
    def __init__(self, filename):
        self.filename = self.trans_path(filename)

    def trans_path(self, filename):
        return "s3://minedojo/" + filename

    def open_mp4(self, start_aug, end_aug):
        video_url = self.filename
        presigned_url = client.generate_presigned_url(video_url, client_method ='get_object', expires_in=3000)
        cap = cv2.VideoCapture(presigned_url)
        fps = cap.get(5)
        frame_count = cap.get(7)
        frames = []
        left = max(int(start_aug * fps), 0)
        # left = 10
        right = min(int(end_aug * fps), frame_count)
        cap.set(1, left)
        while left < right:
            ret, frame = cap.read()
            frames.append(cv2.resize(frame, [640, 360]))
            left += 1
        frames = np.stack(frames)
        cap.release()
        cv2.destroyAllWindows()
        return frames, fps

    def save_mp4(self, video, fps=20):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        url = client.generate_presigned_url(self.filename, client_method='put_object', expires_in=3600)
        out = cv2.VideoWriter(url, fourcc, fps, [640, 360])
        for frame in video:
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()
        raise NotImplementedError

    def save_npy(self, file):
        with io.BytesIO() as f:
            file = np.asanyarray(file)
            np.lib.format.write_array(f, file, allow_pickle=True, pickle_kwargs=dict(fix_imports=True))
            client.put(self.filename, f.getvalue())

    def open_vtt(self):
        if not self.exist():
            return None
        payload = client.get(self.filename)
        try:
            payload = payload.decode("utf-8")
        except Exception as e:
            with open('error.log', 'a') as f1:
                f1.write(self.filename + '\n')
        vtt = webvtt.read_buffer(StringIO(payload))
        # process two lines vtt
        tem = []
        for caption in vtt.captions:
            caption.text = caption.text.split('\n')[-1]
            if len(caption.text) > 1:
                tem.append(caption)
        vtt._captions = tem
        return vtt

    def save_txt(self, file):
        with io.BytesIO() as f:
            f.write(file.encode('utf-8'))
            client.put(self.filename, f.getvalue())

    def save_vtt(self, file):
        writer = webvtt.writers.WebVTTWriter()
        content = writer.webvtt_content(file._captions)
        with io.BytesIO() as f:
            f.write(content.encode('utf-8'))
            client.put(self.filename, f.getvalue())

    def exist(self):
        return client.contains(self.filename)

def takeSecond(elem):
    return elem[1]

def load_video_index(video_index_file, key_word_order=0):
    with open(video_index_file) as f:
        data = json.load(f)
    # data = {'stone': [['--_q5Yp7HGE', 130, 134], ['--6Pu8tq9vk', 145, 149], ['--_q5Yp7HGE', 230, 234],
    #                   ['--_q5Yp7HGE', 330, 334]]
    #     , 'sheep': [['--7kEVKBY8w', 130, 134]]}
    # data = {'stone': [['--_q5Yp7HGE', 1, 3]]}
    ret = []
    t = sorted([(key, len(value)) for key, value in data.items()], key=takeSecond, reverse=True)
    key = t[key_word_order][0]
    for data_name, start, end in data[key]:
        ret.append((key, data_name, start, end))
    ret_dict = {}
    for item in ret:
        if len(ret_dict.get(item[1], [])) > 0:
            ret_dict[item[1]].append(item)
        else:
            ret_dict[item[1]] = [item]
    return ret_dict

def get_video(data_name, start_aug, end_aug, frame_count=64):
    file = File(data_name + '.mp4')
    out_video, _ = file.open_mp4(start_aug, end_aug)
    if len(out_video)>=frame_count:
        index = np.linspace(0, len(out_video) - 1, num=frame_count)
        index = list(map(round, index))
        out_video = out_video[index]
    return out_video

def get_action(data_video, raw_fps, fps):
    # transfer to 20 fps and change raw video
    index = np.linspace(0, len(data_video) - 1, num=int(len(data_video) * fps / raw_fps))
    index = list(map(round, index))
    data_video = data_video[index]

    return data_video

def get_raw_vtt(data_name):
    file = File(data_name + '.en.vtt')
    return file.open_vtt()

def process_vtt(raw_vtt):
    word_bag = []
    for caption in raw_vtt.captions:
        start = caption.start_in_seconds
        end = caption.end_in_seconds
        text = caption.text.split(' ')
        for i, t in enumerate(text):
            s = start + i * (end - start) / len(text)
            e = start + (i + 1) * (end - start) / len(text)
            word_bag.append((s, e, t))
    return word_bag

def get_vtt(word_bag, start_find, end_find, mid):
    start = [i[0] for i in word_bag]
    end = [i[1] for i in word_bag]
    start_index = max(bisect.bisect_left(start, start_find) - 1, 0)
    end_index = min(bisect.bisect_left(end, end_find), len(end))
    assert start_index <= end_index, f'{start_index},{end_index},{start_find},{end_find}'
    ret = ''.join([f'{round(word_bag[i][0]-mid,2)} | {round(word_bag[i][1]-mid,2)} | {word_bag[i][2]}\n' for i in range(start_index, end_index)])
    return ret

def data_save(data_name, data_video, data_txt):
    file = File(data_name + '.npy')
    file.save_npy(data_video)
    file = File(data_name + '.txt')
    file.save_txt(data_txt)


def preprocess_minedojo_videotext_data(args):
    # window=60s
    args, (data_name, video) = args
    video_file = File('videos/' + data_name + '.mp4')
    if not video_file.exist():
        # print(f'{data_name} does not exist')
        return 0
    raw_vtt = get_raw_vtt('videos/' + data_name)
    if raw_vtt is None:
        # print(f'{data_name} does not exist')
        return 0
    word_bag = process_vtt(raw_vtt)
    for i, (keyword, data_name, start, end) in enumerate(video):
        mid = (start + end) / 2
        start_aug = mid - (args.before_time + args.half_video_time)
        end_aug = mid + (args.past_time + args.half_video_time)
        data_video = get_video('videos/' + data_name, mid - args.half_video_time, mid + args.half_video_time, args.n_frames)
        data_txt = get_vtt(word_bag, start_aug, end_aug, mid)
        # data_save('trans/v1' + '/' + keyword + '/' + data_name + '_' + str(i), data_video, data_txt)
        data_save(args.output_path + '/' + keyword + '/' + data_name + '_' + str(i), data_video, data_txt)
    return len(video)

import glob

if __name__ == '__main__':
    path = "/mnt/nfs/wangyuanfu/testdata/*/*.npy"
    out_path = "/mnt/nfs/wangyuanfu/testdata/test/"
    for name in glob.glob(path):
        print(name)
        # load text
        txt_path = name[:-4] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                text = f.read()
                splits = [item.split("|") for item in text.replace(" ", "").split("\n")]
                err = [x for x in splits[:-1] if len(x) != 3]
                if len(err) > 0:
                    print(err)
                splits = [x for x in splits if len(x) == 3]
                captions = {
                    "start": np.array([float(x[0])for x in splits], dtype=np.float16),
                    "end": np.array([float(x[1])for x in splits], dtype=np.float16),
                    "word": np.array([x[2]for x in splits]),
                }
                masked = (captions["start"] >= -8) & (captions["start"] <= 8)
                text = " ".join(captions["word"][masked])
        else:
            continue

        # load video
        frames = np.load(name)
        result = cv2.VideoWriter(f"{name[:-4]}.avi",
            cv2.VideoWriter_fourcc(*'MJPG'),
            4, frames.shape[1:3][::-1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        for frame in frames:
            for i in range(len(text) // 40):
                cv2.putText(frame, 
                    "{}".format(text[i * 40 : (i + 1) * 40]), 
                    (50, 50 + 30 * i), 
                    font, 0.6, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_4)
            result.write(frame)
                
        # release the cap object
        result.release()
        # close all windows
        cv2.destroyAllWindows()

        

        print("Succeed")