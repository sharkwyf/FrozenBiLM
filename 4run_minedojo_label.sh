unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

python demo_video_label.py \
    --combine_datasets none --combine_datasets_val none \
    --video_path="./data/Minedojo/demo.mp4" \
    --output_dir="./data/Minedojo/output/" \
    --model_path="./data/Minedojo/attn.pth" \
    --half_precision=True \
    --load=./output/minedojo/checkpoint0001.pth \
    --answer_bias_weight=100 \
    --max_feats=16 \