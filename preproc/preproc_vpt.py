import os
import sys
import numpy as np

from multiprocessing import Pool
from tqdm import *

workdir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, workdir)
from util.pertrel_oss_helper import PetrelClient

# to read ceph data, MUST run following cmd:
'''
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
'''
 
VPT_CEPH_ROOT_PATH = 's3://vpt/videos/'
VPT_6xx = 'all_6xx_Jun_29/'
VPT_7xx = 'all_7xx_Apr_6/'
VPT_8xx = 'all_8xx_Jun_29/'
VPT_9xx = 'all_9xx_Jun_29/'
VPT_10xx = 'all_10xx_Jun_29/'

SAVE_DATASET_PATH = '/FrozenBiLM/data/VPT/'

cli = PetrelClient()

# select_dir = MULTI_VERSION_LIST[4]

def read_file(video_index):
    # remote video index in petrelOOS
    statistics_dict = {'FILENAME_ERROR': 0,
                       'FILENAME_NOT_EUQAL': 0,
                       'TOO_SHORT': 0,
                       'LENGTH_NOT_EUQAL': 0,
                       'READ_VIDEO_ERROR': 0,
                       'UNKNOW_ERROR': 0,
                       'RIGHT_SAMPLE': 0}

    remote_video_dir = VPT_CEPH_ROOT_PATH + select_dir + video_index  
    meta_info = cli.traverse_dir(remote_video_dir)
    action_meta_file, video_meta_file = '', ''
    for meta_file in meta_info:
        if meta_file[-6:] == '.jsonl':
            action_meta_file = meta_file
        elif meta_file[-4:] == '.mp4':
            video_meta_file = meta_file
    # No such action file or video file!
    if action_meta_file == '' or video_meta_file == '':
        statistics_dict['FILENAME_ERROR'] = 1
        print("No .jsonl or .mp4 file!")
        return 0
    # action file is not same with video file!
    if action_meta_file[:-6] != video_meta_file[:-4]:
        statistics_dict['FILENAME_NOT_EUQAL'] = 1
        print("Video filename not equal to action filename")
        return 0
    
    try:
        # read action info from .jsonl file
        # print("Reading Jsonl: {}{}".format(remote_video_dir, action_meta_file))
        action_info = cli.load_json(remote_video_dir + action_meta_file)
        action_frame_num = len(action_info)
        if action_frame_num < 100:  # action sequence is too stort < 5s
            statistics_dict['TOO_SHORT'] = 1
            print("TOO Short action number!")
            return 0
        read_inventory_info(action_info)
        
        ############## read video info from .mp4 file #######################
        # print("Reading Video: {}{}".format(remote_video_dir, video_meta_file))
        video_stream  = cli.load_video(remote_video_dir + video_meta_file)
        fps, (width, height), frame_count = video_stream.get(5), \
                                        (int(video_stream.get(3)), int(video_stream.get(4))), \
                                        int(video_stream.get(7))
        
        if action_frame_num != frame_count:
            statistics_dict['LENGTH_NOT_EUQAL'] = 1
            print("Action lenght NOT EQUAL to video frame Count!")
            return 0
        
        video_frame_list = []
        # set save path.
        save_video_dir = SAVE_DATASET_PATH + select_dir + video_index
        os.makedirs(save_video_dir, exist_ok=True)
        for _ in range(frame_count):
            success, video_frame = video_stream.read()
            video_frame_list.append(video_frame)
            # image_name = '{}{}.png'.format(save_video_dir, i)
            # cv2.imwrite(image_name, video_frame)
            if not success:
                statistics_dict['READ_VIDEO_ERROR'] = 1
                print("Read video throught cv2.VideoCapture error!")
                return 0
        ####################################################################
        statistics_dict['RIGHT_SAMPLE'] = 1
    except:
        statistics_dict['UNKNOW_ERROR'] = 1
        print("UnKnow error !!!")
        return 0

def read_inventory_info(action_info):
    action_count = len(action_info)
    total_changement_inventory_dict = {} 
    used_inventory_list = []
    for i in range(action_count):
        inventory_info = action_info[i]['inventory']
        for j in range(len(inventory_info)):
            if inventory_info[j]['type'] not in total_changement_inventory_dict.keys():
                total_changement_inventory_dict[inventory_info[j]['type']] = np.zeros(action_count)
            total_changement_inventory_dict[inventory_info[j]['type']][i] = inventory_info[j]['quantity']

    # Check item in inventory change or not.
    for item in total_changement_inventory_dict.keys():
        item_quantity_changement = total_changement_inventory_dict[item]
        changement = item_quantity_changement[1:] - item_quantity_changement[:-1]
        if np.array(changement).any():
            used_inventory_list.append(item)
        changement_index = np.nonzero(changement)
        # if changement_index:
        # from IPython import embed; embed()
            # pass
        
        
    
    return total_changement_inventory_dict.keys(), used_inventory_list

# read_file(video_index_list[0])

def parallel_read_file(select_dir, process_num=100):
    print('Reading DATASET: {}... '.format(VPT_CEPH_ROOT_PATH + select_dir))
    video_index_list = cli.traverse_dir(VPT_CEPH_ROOT_PATH + select_dir)
    print('TOTAL SAMPLE number: {}'.format(len(video_index_list)))
    read_file(video_index_list[1])
    
    # with Pool(processes=process_num) as pool:
    #     results = list(tqdm(
    #         pool.imap_unordered(
    #             read_file,
    #             video_index_list,
    #         ),
    #         total=len(video_index_list)
    #     ))
        
        ##### statistics
        # total_statistics_dict = {'FILENAME_ERROR': 0,
        #                'FILENAME_NOT_EUQAL': 0,
        #                'TOO_SHORT': 0,
        #                'LENGTH_NOT_EUQAL': 0,
        #                'READ_VIDEO_ERROR': 0,
        #                'UNKNOW_ERROR': 0,
        #                'RIGHT_SAMPLE': 0
        #                }
        # for statistics_info in results:
        #     for error_type in statistics_info.keys():
        #         total_statistics_dict[error_type] += statistics_info[error_type]
        
        
        
        # total_items, used_items = [], []
        # error_file_num = 0
        # for result in results:
        #     if result == 0 :
        #         error_file_num += 1
        #         continue
        #     else:
        #         # print(result.keys())
        #         for item in result[0]:
        #             if item not in total_items:
        #                 total_items.append(item)
        #         for item in result[1]:
        #             if item not in used_items:
        #                 used_items.append(item)
    # print('inventory item number (TOTAL): {}'.format(len(total_items)))
    # print('inventory item number (USED): {}'.format(len(used_items)))
    # print('error file number: {} / {}'.format(error_file_num, len(results)))
    # return used_items

all_version_items = []
# MULTI_VERSION_LIST= [VPT_6xx, VPT_7xx, VPT_8xx, VPT_9xx, VPT_10xx ]
MULTI_VERSION_LIST= [VPT_8xx]
for select_dir in MULTI_VERSION_LIST:
    total_items = parallel_read_file(select_dir, process_num=1)
    # for items in total_items:
    #     if items not in all_version_items:
    #         all_version_items.append(items)
# print("ALL VERSION ITEMS: ", all_version_items)
# print("ALL VERSION ITEMS NUMBER: ", len(all_version_items))

