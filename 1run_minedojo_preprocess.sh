unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

python preproc/preproc_minedojo.py \
    --video_index_file='./data/Minedojo/minedojo_clips.json' \
    --output_path='trans/v2' \
    --n_process=16 \