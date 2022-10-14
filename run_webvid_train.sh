python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py \
--combine_datasets webvid --combine_datasets_val webvid --save_dir=trainwebvid \
--lr=3e-5 --ds_factor_ff=8 --ds_factor_attn=8 \
--batch_size=16 --batch_size_val=16 --epochs=2