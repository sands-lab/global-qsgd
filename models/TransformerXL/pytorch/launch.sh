#!/bin/bash
mkdir -p logs
export NCCL_P2P_DISABLE=0
echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
workers=4
rank=32
for seed in 666
do
    # for method in "default" "qsgd" "standard_dithering" "exponential_dithering" "powersgd" "lgreco" "standard_dithering_4bit"
    for method in "standard_dithering_4bit"
    do
        # echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE" > logs/${method}_${seed}.txt
        bash run_wt103_base.sh train $workers $rank TF32/DP32_$seed $seed $method --config dgxa100_4gpu_tf32 >> logs/${method}_${seed}.txt 2>&1
    done
done
