unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

python extract/extract_mineclip_features.py \
    --input_path="s3://minedojo/trans/v2/10000/" \
    --output_path="s3://minedojo/feats/v2/10000/" \
    --model_path="./data/Minedojo/attn.pth" \
    --n_producer1=64 \
    --n_consumer1=8 \
    --n_consumer2=32 \