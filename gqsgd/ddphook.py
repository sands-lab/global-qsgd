
from bz2 import compress
from math import *
from audioop import mul
from threading import local
from tokenize import group
import torch
import torch.distributed as dist
import gqsgd_cuda
from . import allreduce
def standard_dithering_unsigned_deterministic_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    # Allreduce Min & Max
    minmax = torch.stack((-bucket.buffer().min(),bucket.buffer().max()))
    dist.all_reduce(tensor = minmax, op=dist.ReduceOp.MAX, group = group_to_use, async_op=False)
    minimum = -minmax[0]
    maximum = minmax[1] - minimum
     
    # Normalise to [0-1]: Y = (X-MIN)/(MAX-MIN)
    # Compress to [0...254]
    interval = 1/255
    compressed_tensor = bucket.buffer().sub_(minimum).div_(world_size*maximum*interval).round().to(torch.uint8)
    fut = dist.all_reduce(
        compressed_tensor, group=group_to_use, async_op=True
    ).get_future()

    def decompress(fut):
        bucket.set_buffer(fut.value()[0].to(torch.float32).mul_(maximum*interval).add_(minimum))
        return bucket.buffer()

    return fut.then(decompress)

def standard_dithering_deterministic_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # Allreduce Max of abs
    maximum = bucket.buffer().abs().max()
    dist.all_reduce(tensor = maximum, op=dist.ReduceOp.MAX, group = group_to_use, async_op=False)

    # Compress to [-127,127]:
    interval = 1/127
    compressed_tensor = bucket.buffer().div_(world_size*maximum*interval).round().to(torch.int8)
    fut = dist.all_reduce(
        compressed_tensor, group=group_to_use, async_op=True
    ).get_future()

    def decompress(fut):
        bucket.set_buffer(fut.value()[0].to(torch.float32).mul_(maximum*interval))
        return bucket.buffer()

    return fut.then(decompress)

def standard_dithering_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    maximum = bucket.buffer().abs().max()
    dist.all_reduce(tensor = maximum, op=dist.ReduceOp.MAX, group = group_to_use, async_op=False)  # Allreduce Max of abs
    interval = 1/127 
    compressed_tensor = bucket.buffer().div_(world_size*maximum*interval) # Compress to [-127,127]
    gqsgd_cuda.standard_dithering_random_round(compressed_tensor)
    compressed_tensor = compressed_tensor.to(torch.int8)

    #### Customized Allreduce ####
    allreduce.standard_dithering_allreduce(tensor = compressed_tensor)
    decompressed_tensor = compressed_tensor.to(torch.float32).mul_(maximum*interval)
    bucket.set_buffer(decompressed_tensor)
    fut = torch.futures.Future()
    fut.set_result(decompressed_tensor)
    return fut

def exponential_dithering_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    maximum = bucket.buffer().abs().max()
    dist.all_reduce(tensor = maximum, op=dist.ReduceOp.MAX, group = group_to_use, async_op=False)  # Allreduce Max of abs
    maximum *= world_size
    compressed_tensor = gqsgd_cuda.exponential_dithering_compress(bucket.buffer(), maximum)
    allreduce.exponential_dithering_allreduce(tensor = compressed_tensor)
    decompressed_tensor = gqsgd_cuda.exponential_dithering_decompress(compressed_tensor, maximum, world_size)
    bucket.set_buffer(decompressed_tensor)
    fut = torch.futures.Future()
    fut.set_result(decompressed_tensor)
    return fut

# def exponential_dithering_hook(
#     process_group: dist.ProcessGroup, bucket: dist.GradBucket
# ) -> torch.futures.Future[torch.Tensor]:
#     interval = 1/127
#     group_to_use = process_group if process_group is not None else dist.group.WORLD
#     world_size = group_to_use.size()
#     maximum = bucket.buffer().abs().max()
#     compressed_tensor = bucket.buffer()
#     sign = compressed_tensor.sign()
#     compressed_tensor = compressed_tensor.abs()
#     compressed_tensor = compressed_tensor.div_(world_size*maximum)
#     compressed_tensor =compressed_tensor.div_(interval)
#     gqsgd_cuda.standard_dithering_random_round(compressed_tensor)

#     compressed_tensor = compressed_tensor.to(torch.int8)
    
#     #### Customized Allreduce ####
#     # allreduce.standard_dithering_allreduce(tensor = compressed_tensor)
#     # compressed_tensor  = compressed_tensor * sign
#     decompressed_tensor = compressed_tensor.to(torch.float32).mul(interval).mul(maximum*world_size)
#     decompressed_tensor = decompressed_tensor * sign
#     bucket.set_buffer(decompressed_tensor)
#     fut = torch.futures.Future()
#     fut.set_result(decompressed_tensor)
#     return fut
    
    # group_to_use = process_group if process_group is not None else dist.group.WORLD
    # # world_size = group_to_use.size()
    # maximum = bucket.buffer().abs().max()
    # # dist.all_reduce(tensor = maximum, op=dist.ReduceOp.MAX, group = group_to_use, async_op=False)  # Allreduce Max of abs
    # rank = dist.get_rank()
    # compressed_tensor = bucket.buffer()
    # # if rank == 0:
    # #     print("Orginal tensor:", bucket.buffer())

    # compressed_tensor = compressed_tensor.abs().div(maximum)
    # # if rank == 0:
    # #     print("normalized tensor:", compressed_tensor)
    # compressed_tensor = torch.log(compressed_tensor)/torch.log(torch.tensor(2.0))
    # gqsgd_cuda.standard_dithering_random_round(compressed_tensor)
    # compressed_tensor = compressed_tensor.to(torch.int8)
    # # if rank == 0:
    # #     print("compressed tensor:", compressed_tensor)
    # # print(compressed_tensor)
    # decompressed_tensor = torch.pow(torch.tensor(2.0), compressed_tensor.to(torch.float32))
    # decompressed_tensor = decompressed_tensor.mul(maximum)
    # decompressed_tensor = decompressed_tensor.mul(sign)
    # # if rank == 0:
    # #     print("decompressed tensor:", decompressed_tensor)
    # # print("Orginal tensor:", bucket.buffer())
    # # print("Compressed tensor:", compressed_tensor)
    # # print("Decompressed tensor:", decompressed_tensor)
    # bucket.set_buffer(decompressed_tensor)
    # fut = torch.futures.Future()
    # fut.set_result(compressed_tensor)
    # return fut