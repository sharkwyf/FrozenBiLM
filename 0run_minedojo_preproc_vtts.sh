unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

python preproc/preproc_minedojo_vtts.py \
    --input_path="s3://minedojo/videos/" \
    --keyword_path="/FrozenBiLM/data/Minedojo/keywords.json" \
    --output_path="/FrozenBiLM/data/Minedojo/minedojo_clips.json" \
    --n_process=96 \