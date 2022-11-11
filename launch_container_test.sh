FRAME_SET='/mnt/nfs2/yangchao/1_VL/EgoVLP/EgoVLP/part8'

docker run --gpus all --ipc=host --network=bridge --expose 80 --rm -itd\
    --mount src=$(pwd),dst=/FrozenBiLM,type=bind \
    --mount src=$FRAME_SET,dst=/DATASET,type=bind \
    --env HTTP_PROXY="http://10.1.8.5:32680/" \
    --env HTTPS_PROXY="http://10.1.8.5:32680/" \
    -w /FrozenBiLM yangchao_test:latest \
    bash -c "bash" 

    # cd /FrozenBiLM
    # ./run_minedojo_train.sh
