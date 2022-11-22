# Pretrain
export http_proxy=http://10.1.8.5:32680/
export https_proxy=http://10.1.8.5:32680/
export HTTP_PROXY=http://10.1.8.5:32680/
export HTTPS_PROXY=http://10.1.8.5:32680/
# export no_proxy=http://10.140.2.204,http://10.140.14.204
# export no_proxy="`echo 10.140.{1..255}.{1..254},`"


# pip install wandb
wandb offline
cp /FrozenBiLM/repo/.netrc ~/

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py \
    --combine_datasets minedojo \
    --combine_datasets_val minedojo \
    --lr=3e-5 --ds_factor_ff=8 --ds_factor_attn=8 \
    --batch_size=64 --batch_size_val=64 \
    --max_feats=16 --features_dim=768 \
    --minedojo_mask_probs 0. 0.15 0. \
    --minedojo_loss_weights 0 1 0 \
    --video_index_file=/FrozenBiLM/data/Minedojo/minedojo_clips_v2.json \
    --minedojo_features_path=s3://minedojo/feats/v2/15000/ \
    --epochs=40 \
    --eval_skip=4 \
    --minedojo_text_max_range -2 2 \
    --minedojo_vid_min_range -2 2 \
    --minedojo_vid_max_range -4 4 \
    --word_mask_probs 0.15 0.15 \
    --save_dir=/FrozenBiLM/output/v2/v48t4_n15v15 \
    --load=/FrozenBiLM/checkpoints/frozenbilm.pth \
    # --load=/FrozenBiLM/output/v2/v6t4_n15v15/checkpoint0019.pth \
    # --resume \
