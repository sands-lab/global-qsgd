import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import gqsgd_cuda
from gqsgd import allreduce

def run(rank, size):
    repeat = 1
    # accum =torch.tensor([0],dtype=float).cuda(rank)
    # for i in range(repeat):
    #     if rank ==0 or rank ==1:
    #         input = torch.tensor([0.008]).cuda(rank)
    #     else:
    #         input = torch.tensor([0.010]).cuda(rank)
    #     global_norm = torch.tensor([1],dtype=torch.float).cuda(rank)
    #     compressed = gqsgd_cuda.exponential_dithering_compress(input, global_norm)
    #     allreduce.tree_allreduce(tensor = compressed, exponential = True)
    #     print("reduced value: ", compressed)
    #     decompressed = gqsgd_cuda.exponential_dithering_decompress(compressed, global_norm,size)
    #     accum += decompressed
    # print(accum/repeat)

    for i in range(repeat):
        input = torch.tensor([[rank]*2]*2).cuda(rank)
        tensor_list = allreduce.tree_allgather(tensor = input)
        print("Rank:",rank, "", tensor_list)


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()