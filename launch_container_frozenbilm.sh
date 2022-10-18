
docker run --gpus all --ipc=host --network=bridge --expose 80 --rm -itd\
    --mount src=$(pwd),dst=/FrozenBiLM,type=bind \
    --env DATA_DIR="/FrozenBiLM/data" \
    --env TRANSFORMERS_CACHE="/FrozenBiLM/transformers_cache" \
    --env HTTP_PROXY="http://10.1.8.5:32680/" \
    --env HTTPS_PROXY="http://10.1.8.5:32680/" \
    -w /FrozenBiLM sharkwyf/frozenbilm:latest \
    bash -c "bash" 

    # cd /FrozenBiLM
    # ./run_minedojo_train.sh
