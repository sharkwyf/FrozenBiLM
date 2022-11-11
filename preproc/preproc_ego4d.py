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
 
EGO4D_CEPH_ROOT_PATH = 's3://ego4d/ego4d_frame/'

cli = PetrelClient()

LOCAL_PATH = '/DATASET/'


def upload_file(caption_index):
    remote_url = EGO4D_CEPH_ROOT_PATH + caption_index
    local_dir = LOCAL_PATH + caption_index
    cli.save_image(local_dir, remote_url)
    
    

def parallel_upload_file(LOCAL_PATH, process_num=100):
    
    print('Reading DATASET: {}... '.format(LOCAL_PATH))
    caption_index_list = os.listdir(LOCAL_PATH)
    
    
    print('TOTAL SAMPLE number: {}'.format(len(caption_index_list)))
    
    with Pool(processes=process_num) as pool:
        results = list(tqdm(
            pool.imap_unordered(
                upload_file,
                caption_index_list,
            ),
            total=len(caption_index_list)
        ))

parallel_upload_file(LOCAL_PATH, process_num=100)
