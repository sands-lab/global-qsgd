seeds=(100)
for seed in ${seeds[@]}
do
mpirun -np 4 python main_all.py -l2 6e-7 -n_epochs 10 -warm 2 -prune 1 -sparse 0.90 -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1. -batch_size 512 -backend=nccl -random_seed=$seed -hook=default > logs/PCIe/log_default_seed$seed.txt 2>&1
# mpirun -np 4 python main_all.py -l2 6e-7 -n_epochs 10 -warm 2 -prune 1 -sparse 0.90 -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1. -batch_size 512 -backend=nccl -random_seed=$seed -hook=standard_dithering > logs/PCIe/log_standard_seed$seed.txt 2>&1
# mpirun -np 4 python main_all.py -l2 6e-7 -n_epochs 10 -warm 2 -prune 1 -sparse 0.90 -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1. -batch_size 512 -backend=nccl -random_seed=$seed -hook=exponential_dithering > logs/PCIe/log_exponential_seed$seed.txt 2>&1
# mpirun -np 4 python main_all.py -l2 6e-7 -n_epochs 10 -warm 2 -prune 1 -sparse 0.90 -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1. -batch_size 512 -backend=nccl -random_seed=$seed -hook=lgreco > logs/log_lgreco_seed$seed.txt 2>&1
#  mpirun -np 4 python main_all.py -l2 6e-7 -n_epochs 10 -warm 2 -prune 1 -sparse 0.90 -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1. -batch_size 512 -backend=nccl  -random_seed=$seed -hook=qsgd > logs/log_qsgd_seed$seed.txt 2>&1
done