unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

python extract/extract_mineclip_features.py \
    --trans="s3://minedojo/trans/v1/" \
    --output_path="./data/Minedojo/mineclip_features.npy" \
    --half_precision=True \
    --world_size=8 \
    # --keywords stone_pickaxe \