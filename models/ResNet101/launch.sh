seeds=(666)
DATA_DIR=/root/miniimagenet
export NCCL_P2P_DISABLE=0
echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
for seed in ${seeds[@]}
do
# echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE" > logs/default_seed$seed.txt 2>&1
# python3 main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook default $DATA_DIR >> logs/default_seed$seed.txt 2>&1
# python3 main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook qsgd $DATA_DIR >logs/qsgd_seed$seed.txt 2>&1
# echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE" > logs/standard_seed$seed.txt 2>&1
# python3 main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook standard_dithering $DATA_DIR >logs/standard_seed$seed.txt 2>&1

# echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE" > logs/exponential_seed$seed.txt 2>&1
# python3 main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook exponential_dithering $DATA_DIR >logs/exponential_seed$seed.txt 2>&1
# python3 main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook powersgd $DATA_DIR >logs/powersgd_seed$seed.txt 2>&1
# python3 main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook lgreco $DATA_DIR >logs/lgreco_seed$seed.txt 2>&1

echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE" > logs/standard_dithering_4bit_seed$seed.txt 2>&1
python3 main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook standard_dithering_4bit $DATA_DIR >> logs/standard_dithering_4bit_seed$seed.txt 2>&1
done
