import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import gqsgd_cuda
from gqsgd import allreduce

def run(rank, size):
    repeat = 1000
    accum =torch.tensor([0],dtype=float).cuda(rank)
    for i in range(repeat):
        if rank ==0:
            # input = torch.tensor([-8,-4,-2,-1,0,1,2,4,8],dtype=torch.float).cuda(rank)
            input = torch.tensor([0.5]).cuda(rank)
        else:
            # input = torch.tensor([8,4,2,1,0,-1,-2,-4,-8],dtype=torch.float).cuda(rank)
            input = torch.tensor([-0.3]).cuda(rank)
        # if rank ==0:
        #     print("rank ", rank, " Original Value:")
        #     print(input)
        #     print("")
        global_norm = torch.tensor([1],dtype=torch.float).cuda(rank)
        # if rank ==0:
        #     print("Normalized Value:")
        #     print(input/(global_norm))
        #     print("")
        compressed = gqsgd_cuda.exponential_dithering_compress(input, global_norm)
        # interval = 1/127 
        # compressed = input.div_(size*global_norm*interval) # Compress to [-127,127]
        # gqsgd_cuda.standard_dithering_random_round(compressed)
        # if rank ==0:
        # print("Compressed Value:")
        # print(compressed)
        # print("")
        allreduce.tree_allreduce(tensor = compressed, exponential = True)
        # if rank ==0:
        #     print("Reduced Compressed Value:")
        #     print(compressed)
        #     print("")
        decompressed = gqsgd_cuda.exponential_dithering_decompress(compressed, global_norm,size)
        # decompressed = compressed.to(torch.float32).mul_(global_norm*interval)
        accum += decompressed
        # if rank ==0:
        #     print("Decompressed Value:")
        #     print(decompressed)
        #     print("")
    print(accum/repeat)


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

