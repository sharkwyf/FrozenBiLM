import os
import sys
from turtle import width
import cv2 
import numpy as np
from multiprocessing import Pool
from multiprocessing import set_start_method
from tqdm import *
workdir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, workdir)
from util.pertrel_oss_helper import PetrelClient
from repo.VPT.vpt_data_process.generate_data_inverse_dynamics_model import Agent

# to read ceph data, MUST run following cmd:
# cmd = "unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY"

HUMAN_LABELED_VPT_PATH = 's3://vpt/videos/'
VPT_6xx = 'all_6xx_Jun_29/'
VPT_7xx = 'all_7xx_Apr_6/'
VPT_8xx = 'all_8xx_Jun_29/'
VPT_9xx = 'all_9xx_Jun_29/'
VPT_10xx = 'all_10xx_Jun_29/'
BASALT_waterfall = 'waterfall-Jul-28/'
BASALT_findcave = 'find-cave-Jul-28/'
BASALT_pen_animals = 'pen-animals-Jul-28/'
BASALT_buildhouse = 'build-house-Jul-28/'

SAVE_DATASET_PATH = '/FrozenBiLM/data/VPT/'

MULTI_VERSION_LIST= [VPT_6xx, 
    VPT_7xx,
    VPT_8xx,
    VPT_9xx,
    VPT_10xx,
    BASALT_waterfall,
    BASALT_findcave,
    BASALT_pen_animals,
    BASALT_buildhouse
    ]

cli = PetrelClient()


model_file = '/FrozenBiLM/repo/VPT/model_weight_files/4x_idm.model'
weights_file = '/FrozenBiLM/repo/VPT/model_weight_files/4x_idm.weights'

idm_agent = Agent(model=model_file, 
                  weights=weights_file, 
                  n_frames=128, device='cuda:4')

NULL_ACTION= {}
NULL_ACTION['buttons'] = 0
NULL_ACTION['camera'] = 60

SAVE_PATH = 's3://basalt/bc_data/v0/'

def read_file(video_index):
    CEPH_URL_PATH = HUMAN_LABELED_VPT_PATH + select_dir
    video_path = CEPH_URL_PATH + video_index
    meta_info = cli.traverse_dir(video_path)
    action_meta_file, video_meta_file = '', ''
    for meta_file in meta_info:
        if meta_file[-6:] == '.jsonl':
            action_meta_file = meta_file
        elif meta_file[-4:] == '.mp4':
            video_meta_file = meta_file
    # No such action file or video file!
    if action_meta_file == '' or video_meta_file == '':
        return 0
    # action file is not same with video file!
    if action_meta_file[:-6] != video_meta_file[:-4]:
        return 0
    
    # try:
    # read action info from .jsonl file
    action_info = cli.load_json(video_path + action_meta_file)
    action_frame_num = len(action_info)

    ############ read video info from .mp4 file #######################
    print(video_meta_file)
    video_stream  = cli.load_video(video_path + video_meta_file)
    fps, (width, height), frame_count = video_stream.get(5), \
                                        (int(video_stream.get(3)), int(video_stream.get(4))), \
                                        int(video_stream.get(7))
    if frame_count < 128:  # action sequence is too stort < 5s
        print("Too short video!!")
        return 0
    video_frame_list = []
    save_video_dir = SAVE_DATASET_PATH + select_dir + video_index  # set save path.
    os.makedirs(save_video_dir, exist_ok=True)
    for i in range(frame_count):
        success, video_frame = video_stream.read()
        video_frame_list.append(video_frame)
        # image_name = '{}{}.png'.format(save_video_dir, i)
        # cv2.imwrite(image_name, video_frame)
        if not success:
            return 0
    
    # prediciton action from video frames 
    video_frame = np.stack(video_frame_list)    
    predict_action_npy = idm_agent.predict(video_frame)
    
    # remove null action and corresponding video frame
    filtered_index = []
    for i in range(predict_action_npy['buttons'].shape[0]):
        if predict_action_npy['buttons'][i] != NULL_ACTION['buttons'] & \
            predict_action_npy['camera'][i] != NULL_ACTION['camera']:
            filtered_index.append(i)
    filtered_action = {}
    filtered_action['buttons'] = predict_action_npy['buttons'][filtered_index]
    filtered_action['camera'] = predict_action_npy['camera'][filtered_index]
    
    video_frame = video_frame[filtered_index]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # local_video_filename = './' + video_meta_file
    
    videoWriter = cv2.VideoWriter(video_meta_file, 
                                  fourcc, fps, (width, height))
    for j in range(len(video_frame)):
        videoWriter.write(video_frame[j])
    videoWriter.release()
    
    target_ceph_dir = SAVE_PATH + select_dir + video_index
    cli.save_video(video_meta_file, target_ceph_dir + video_meta_file, target_ceph_dir)
    
    npy_file = video_meta_file[:-4] + '.npy'
    cli.save_npy_direct('{}{}'.format(target_ceph_dir, npy_file), np.array(filtered_action))
    
    ##################################################################
        # total_inventory_dict = {} 
        # used_inventory_list = []
        # for i in range(action_frame_num):
        #     inventory_info = action_info[i]['inventory']
        #     for j in range(len(inventory_info)):
        #         if inventory_info[j]['type'] not in total_inventory_dict.keys():
        #             total_inventory_dict[inventory_info[j]['type']] = np.zeros(action_frame_num)
        #         total_inventory_dict[inventory_info[j]['type']][i] = inventory_info[j]['quantity']
    #     return action_frame_num
    # except:
    return 0

def parallel_read_file(select_dir, process_num):
    print('Reading DATASET: {}: '.format(select_dir))
    CEPH_URL_PATH = HUMAN_LABELED_VPT_PATH + select_dir
    video_index_list = cli.traverse_dir(CEPH_URL_PATH)
        
    for i in range(len(video_index_list)):
        results = read_file(video_index_list[i])

for select_dir in MULTI_VERSION_LIST[8:9]:
    parallel_read_file(select_dir, process_num=1)
    