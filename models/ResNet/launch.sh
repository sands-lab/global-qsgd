seeds=(666 999)
for seed in ${seeds[@]}
do
python main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook default /home/ubuntu/jihao/data/mini-imagenet >logs/seed$seed/default_seed$seed.txt 2>&1
python main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook qsgd /home/ubuntu/jihao/data/mini-imagenet >logs/seed$seed/qsgd_seed$seed.txt 2>&1
python main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook standard_dithering /home/ubuntu/jihao/data/mini-imagenet >logs/seed$seed/standard_seed$seed.txt 2>&1
python main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook exponential_dithering /home/ubuntu/jihao/data/mini-imagenet >logs/seed$seed/exponential_seed$seed.txt 2>&1
python main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook powersgd /home/ubuntu/jihao/data/mini-imagenet >logs/seed$seed/powersgd_seed$seed.txt 2>&1
python main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook lgreco /home/ubuntu/jihao/data/mini-imagenet >logs/seed$seed/lgreco_seed$seed.txt 2>&1
done

# seeds=(666 999)
# for seed in ${seeds[@]}
# do
# python main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook default /home/ubuntu/jihao/data/mini-imagenet >logs/seed$seed/default_seed$seed.txt 2>&1
# python main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook qsgd /home/ubuntu/jihao/data/mini-imagenet >logs/seed$seed/qsgd_seed$seed.txt 2>&1
# python main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook standard_dithering /home/ubuntu/jihao/data/mini-imagenet >logs/seed$seed/standard_seed$seed.txt 2>&1
# python main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook exponential_dithering /home/ubuntu/jihao/data/mini-imagenet >logs/seed$seed/exponential_seed$seed.txt 2>&1
# python main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook powersgd /home/ubuntu/jihao/data/mini-imagenet >logs/seed$seed/powersgd_seed$seed.txt 2>&1
# python main.py -a wide_resnet101_2 --epochs=90 --batch-size 64 --dist-url 'tcp://127.0.0.1:12701' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --seed=$seed --hook lgreco /home/ubuntu/jihao/data/mini-imagenet >logs/seed$seed/lgreco_seed$seed.txt 2>&1
# done