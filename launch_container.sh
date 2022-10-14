
docker run --gpus all --ipc=host --rm -itd \
    --mount src=$(pwd),dst=/FrozenBiLM,type=bind \
    --env DATA_DIR="/FrozenBiLM/data" \
    --env TRANSFORMERS_CACHE="/FrozenBiLM/transformers_cache" \
    --env HTTP_PROXY="http://10.1.8.5:32680/" \
    --env HTTPS_PROXY="http://10.1.8.5:32680/" \
    -w /FrozenBiLM sharkwyf/frozenbilm \
    bash -c "bash" 
