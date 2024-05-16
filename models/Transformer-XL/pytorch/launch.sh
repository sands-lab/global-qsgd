#!/bin/bash
workers=4
rank=32
for seed in 666
do
    # for method in "default" "qsgd" "standard_dithering" "exponential_dithering" "powersgd" "lgreco"
    for method in "powersgd" "lgreco"
    do
        bash run_wt103_base.sh train $workers $rank TF32/DP32_$seed $seed $method --config dgxa100_4gpu_tf32 > logs/${method}_${seed}.txt 2>&1
    done
done
