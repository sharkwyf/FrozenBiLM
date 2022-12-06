
docker run --gpus all --ipc=host --network=bridge --expose 80 --rm -itd \
    --mount src=$(pwd),dst=/FrozenBiLM,type=bind \
    -w /FrozenBiLM sharkwyf/minedojo:dev \
    bash -c "bash" 
