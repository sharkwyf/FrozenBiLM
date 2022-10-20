unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

python extract/demo_video_label.py \
    --combine_datasets none --combine_datasets_val none \
    --video_path="./data/Minedojo/demo.mp4" \
    --output_path="./data/Minedojo/demo_output.mp4" \
    --model_path="./data/Minedojo/attn.pth" \
    --half_precision=True \
    --load=./output/minedojo/checkpoint0001.pth \