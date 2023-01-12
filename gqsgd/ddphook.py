
from bz2 import compress
from math import *
from audioop import mul
from threading import local
from tokenize import group
import torch
import torch.distributed as dist
import gqsgd_cuda
from . import allreduce

def default_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements standard dithering quantization based on global norm.
    It will quantize the gradient tensor to 8-bit signed integer: Normalize by global norm, multiply by 127 ->[-127,0,127], then round with probability.
    The quantized tensor will be allreduced then dequantized to the same data type as the input gradient tensor.

    Example::
        >>> ddp_model.register_comm_hook(process_group(None), standard_dithering_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    buffer = bucket.buffer().div_(world_size)
    allreduce.tree_allreduce(tensor = compressed_tensor, exponential = False)
    # allreduce.standard_dithering_allreduce(tensor = buffer)
    fut = torch.futures.Future()
    fut.set_result(buffer)
    return fut

def standard_dithering_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements standard dithering quantization based on global norm.
    It will quantize the gradient tensor to 8-bit signed integer: Normalize by global norm, multiply by 127 ->[-127,0,127], then round with probability.
    The quantized tensor will be allreduced then dequantized to the same data type as the input gradient tensor.

    Example::
        >>> ddp_model.register_comm_hook(process_group(None), standard_dithering_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    maximum = bucket.buffer().abs().max()
    dist.all_reduce(tensor = maximum, op=dist.ReduceOp.MAX, group = group_to_use, async_op=False)  # Allreduce Max of abs
    interval = 1/127 
    compressed_tensor = bucket.buffer().div_(world_size*maximum*interval) # Compress to [-127,127]
    gqsgd_cuda.standard_dithering_random_round(compressed_tensor)
    compressed_tensor = compressed_tensor.to(torch.int8)
    allreduce.tree_allreduce(tensor = compressed_tensor, exponential = False)
    decompressed_tensor = compressed_tensor.to(torch.float32).mul_(maximum*interval)
    fut = torch.futures.Future()
    fut.set_result(decompressed_tensor)
    return fut

def exponential_dithering_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements exponential dithering quantization based on global norm.
    It will quantize the gradient tensor to 8-bit unsigned integer: Normalize by global normthen then use CUDA function to compress.
    The quantized tensor will be allreduced with customized function then dequantized to the same data type as the input gradient tensor.

    Example::
        >>> ddp_model.register_comm_hook(process_group(None), exponential_dithering_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    maximum = bucket.buffer().abs().max()
    dist.all_reduce(tensor = maximum, op=dist.ReduceOp.MAX, group = group_to_use, async_op=False)  # Allreduce Max of abs
    maximum *= world_size
    compressed_tensor = gqsgd_cuda.exponential_dithering_compress(bucket.buffer(), maximum)
    allreduce.tree_allreduce(tensor = compressed_tensor, exponential = True)
    decompressed_tensor = gqsgd_cuda.exponential_dithering_decompress(compressed_tensor, maximum, world_size)
    fut = torch.futures.Future()
    fut.set_result(decompressed_tensor)
    return fut