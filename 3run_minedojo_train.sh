unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py \
    --combine_datasets minedojo \
    --combine_datasets_val minedojo \
    --video_index_file=/FrozenBiLM/data/Minedojo/minedojo_clips_full.json \
    --minedojo_features_path=s3://minedojo/feats/v1/ \
    --save_dir=/FrozenBiLM/output/4s \
    --lr=3e-5 --ds_factor_ff=8 --ds_factor_attn=8 \
    --batch_size=64 --batch_size_val=64 --epochs=20 \
    --eval_skip=1 \
    --max_feats=16 --features_dim=768 \
    --minedojo_text_start=-2 \
    --minedojo_text_end=2 \
    --minedojo_vid_start=-2 \
    --minedojo_vid_end=2 \
    --minedojo_mask_probs 0. 0.15 0. \
    --minedojo_loss_weights 0 1 0 \
    --resume \
    --load=/FrozenBiLM/output/4s/checkpoint0011.pth
    # --load=/FrozenBiLM/checkpoints/frozenbilm.pth \
