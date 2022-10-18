
docker run --gpus all --ipc=host --rm -itd \
    --mount src=$(pwd),dst=/FrozenBiLM,type=bind \
    --env DATA_DIR="/FrozenBiLM/data" \
    --env TRANSFORMERS_CACHE="/FrozenBiLM/transformers_cache" \
    -w /FrozenBiLM sharkwyf/mineclip:latest \
    bash -c "unset HTTP_PROXY; unset HTTPS_PROXY; bash" 


    # unset HTTP_PROXY
    # unset HTTPS_PROXY
    # cd /FrozenBiLM/extract
    # python extract_mineclip_features.py --world_size=8
