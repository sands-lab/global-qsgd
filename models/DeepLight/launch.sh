export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export NCCL_P2P_DISABLE=1
echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
for seed in 666
do
    # for method in "default" "qsgd" "standard_dithering" "exponential_dithering" "powersgd" "lgreco" "standard_dithering_4bit"
    for method in "standard_dithering_4bit" "exponential_dithering" "powersgd" "lgreco" 
    do
        echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE" > logs/${method}_${seed}.txt 2>&1
        mpirun -np 4 python3 main_all.py -l2 6e-7 -n_epochs 10 -warm 2 -prune 1 -sparse 0.90 -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1. -batch_size 512 -backend=nccl -random_seed=$seed -hook=$method >> logs/${method}_${seed}.txt 2>&1
    done
done