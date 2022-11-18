unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

python preproc/preproc_minedojo_vids.py \
    --video_index_file='./data/Minedojo/minedojo_clips_v3.json' \
    --input_path='s3://minedojo/videos/' \
    --output_path='s3://minedojo/trans/v3/15000/' \
    --max_clips_per_keyword=15000 \
    --world_size 1 \
    --rank 0 \