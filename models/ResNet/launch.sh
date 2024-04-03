seeds=(666 999)
DATA_DIR=/mnt/xinj/miniimagenet
for seed in ${seeds[@]}
do
python3 main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook default $DATA_DIR >logs/seed$seed/default_seed$seed.txt 2>&1
python3 main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook qsgd $DATA_DIR >logs/seed$seed/qsgd_seed$seed.txt 2>&1
python3 main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook standard_dithering $DATA_DIR >logs/seed$seed/standard_seed$seed.txt 2>&1
python3 main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook exponential_dithering $DATA_DIR >logs/seed$seed/exponential_seed$seed.txt 2>&1
python3 main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook powersgd $DATA_DIR >logs/seed$seed/powersgd_seed$seed.txt 2>&1
python3 main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook lgreco $DATA_DIR >logs/seed$seed/lgreco_seed$seed.txt 2>&1
done
