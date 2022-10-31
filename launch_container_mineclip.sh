
docker run --gpus all --ipc=host --network=bridge --expose 80 --rm -itd --cpus="112" \
    --mount src=$(pwd),dst=/FrozenBiLM,type=bind \
    --env DATA_DIR="/FrozenBiLM/data" \
    --env TRANSFORMERS_CACHE="/FrozenBiLM/transformers_cache" \
    -w /FrozenBiLM sharkwyf/mineclip:latest \
    bash -c "unset HTTP_PROXY; unset HTTPS_PROXY; bash" 


    # unset HTTP_PROXY
    # unset HTTPS_PROXY
    # cd /FrozenBiLM/extract
    # python extract_mineclip_features.py --world_size=8
