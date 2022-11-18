unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
export TRANSFORMERS_CACHE=./transformers_cache/

VERSION=v3

for RUN in v6t3_n60v100 v8t3_n60v100 # v4t4_n15v15 v4t4_n30v30 v4t4_n60v60 v6t4_n100v100    #
do
    echo $RUN
    python benchmark_eval.py \
        --combine_datasets none --combine_datasets_val none \
        --feature_path=./data/Minedojo/benchmarks/features.npy \
        --load=./output/$VERSION/$RUN/checkpoint0000.pth \
        --answer_bias_weight=0
done