unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

python extract/extract_idm_features.py \
    --video_index_file='./data/Minedojo/minedojo_clips_v2.json' \
    --input_path='s3://minedojo/videos/' \
    --output_path='s3://minedojo/idms/v2/15000/' \
    --model='./data/Minedojo/4x_idm.model' \
    --weights='./data/Minedojo/4x_idm.weights' \
    --max_clips_per_keyword=15000 \
    --n_downloader=116 \
    --n_extractor=8 \
    --n_uploader=4 \
    --world_size 2 \
    --rank 0 \
