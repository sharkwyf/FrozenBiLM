python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py \
--combine_datasets minedojo --combine_datasets_val minedojo --save_dir=/FrozenBiLM/output/minedojo \
--lr=3e-5 --ds_factor_ff=8 --ds_factor_attn=8 \
--batch_size=96 --batch_size_val=16 --epochs=3000 \
--eval_skip=120 \
--max_feats=10 --features_dim=768 \
--resume --load=/FrozenBiLM/checkpoints/frozenbilm.pth \