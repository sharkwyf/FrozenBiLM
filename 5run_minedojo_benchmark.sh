unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
export TRANSFORMERS_CACHE=./transformers_cache/

VERSION=v2

for RUN in v6t4_n15v15 # v4t4_n30v30 v4t4_n60v60 v6t4_n100v100 # v8t2_n100v100 # v4t4_n15v15 v4t4_n30v30 v4t4_n60v60 v6t4_n100v100    #
do
    for ckpt in 03 07 11 15 19 23 27 31 35 39
    do
        echo /$VERSION/$RUN/checkpoint00$ckpt.pth
        python benchmark_eval.py \
            --combine_datasets none --combine_datasets_val none \
            --ds_factor_ff=8 --ds_factor_attn=8 \
            --feature_path=./data/Minedojo/benchmarks/features.npy \
            --load=./output/$VERSION/$RUN/checkpoint00$ckpt.pth \
            # --answer_bias_weight=-100
    done
done