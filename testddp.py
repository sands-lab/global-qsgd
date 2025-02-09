import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import gqsgd_cuda
from gqsgd import allreduce
import torch.distributed as dist

def run(rank, world_size):
    torch.cuda.set_device(rank)
    if rank == 0 or rank == 1:
        input = torch.tensor([-10],dtype=torch.float32).cuda(rank)
    else:
        input = torch.tensor([20],dtype=torch.float32).cuda(rank)
    global_norm = torch.tensor([20],dtype=torch.float32).cuda(rank)
    print("Input: ", input)

    # Compress Communication Decompress
    mode = 3 # 1 =  Standard Dithering Allreduce, 2 = Exponential Dithering Allreduce, 3 = Allgather
    if mode == 1: # Standard Dithering Allreduce
        interval = 1/127 
        normalized_tensor = input.div_(world_size*global_norm*interval) # Compress to [-127,127]
        print("Normalized: ", normalized_tensor)
        gqsgd_cuda.standard_dithering_random_round(normalized_tensor)
        print("Rounded: ", normalized_tensor)
        compressed_tensor = normalized_tensor.to(torch.int8)
        print("Compressed to int8: ", compressed_tensor)
        allreduce.tree_allreduce(tensor = compressed_tensor, exponential = False)
        print("Allreduced: ", compressed_tensor)
        decompressed_tensor = compressed_tensor.to(torch.float32).mul_(global_norm*interval)
        print("Decompressed: ", decompressed_tensor)
        if rank == 0:
            print("Standard Dithering Allreduce: ", decompressed_tensor)
    elif mode == 2:
        global_norm *=2*world_size
        compressed = gqsgd_cuda.exponential_dithering_compress(input, global_norm)
        print("Compressed: ", compressed)
        allreduce.tree_allreduce(tensor = compressed, exponential = True)
        print("Allreduced: ", compressed)
        decompressed = gqsgd_cuda.exponential_dithering_decompress(compressed, global_norm,world_size)
        print("Decompressed: ", decompressed)
    elif mode == 3:
        tensor_list = allreduce.tree_allgather(tensor = input)
        print("Allgather: ", tensor_list)


def init_process(rank, world_size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank, world_size)


if __name__ == "__main__":
    world_size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()