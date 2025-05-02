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
    allreduce.tree_allreduce(tensor = buffer, exponential = False)
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

def THC_uniform_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements THC uniform quantization.
    It will quantize the gradient tensor to 8-bit signed integer: Normalize by global norm, multiply by 127 ->[-127,0,127], then round with probability.
    The quantized tensor will be allreduced then dequantized to the same data type as the input gradient tensor.

    Example::
        >>> ddp_model.register_comm_hook(process_group(None), THC_uniform_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    maximum = bucket.buffer().abs().max()
    dist.all_reduce(tensor = maximum, op=dist.ReduceOp.MAX, group = group_to_use, async_op=False)  # Allreduce Max of abs
    interval = 1/127 
    compressed_tensor = bucket.buffer().div_(world_size*maximum*interval) # Compress to [-127,127]
    gqsgd_cuda.standard_dithering_random_round(compressed_tensor)
    compressed_tensor = compressed_tensor.to(torch.int8)
    allreduce.ps_allreduce(tensor = compressed_tensor, exponential = False)
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
    maximum *= 2*world_size # 4 workers, each [0-0.125] -> [0,3-127]
    compressed_tensor = gqsgd_cuda.exponential_dithering_compress(bucket.buffer(), maximum)
    allreduce.tree_allreduce(tensor = compressed_tensor, exponential = True)
    decompressed_tensor = gqsgd_cuda.exponential_dithering_decompress(compressed_tensor, maximum, world_size)
    fut = torch.futures.Future()
    fut.set_result(decompressed_tensor)
    return fut

def qsgd_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements qsgd without elis encoding.
    It will quantize the gradient tensor to 8-bit signed integer: Normalize by local norm, multiply by 127 ->[-127,0,127], then round with probability.
    The quantized tensor will be all_gathered then dequantized to the same data type as the input gradient tensor.

    Example::
        >>> ddp_model.register_comm_hook(process_group(None), standard_dithering_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    maximum = bucket.buffer().abs().max()
    # Allgather maximum
    maximums = allreduce.tree_allgather(maximum)
    interval = 1/127 
    compressed_tensor = bucket.buffer().div_(world_size*maximum*interval) # Compress to [-127,127]
    gqsgd_cuda.standard_dithering_random_round(compressed_tensor)
    compressed_tensor = compressed_tensor.to(torch.int8)
    # Allgather
    tensors = allreduce.tree_allgather(compressed_tensor)
    # Decompress
    decompressed_tensor = torch.zeros_like(bucket.buffer())
    for i in range(world_size):
        decompressed_tensor.add_(tensors[i].to(torch.float32).mul_(maximums[i]*interval))
    fut = torch.futures.Future()
    fut.set_result(decompressed_tensor)
    return fut

def pack_4bits_tensor(tensor):
    packed = torch.zeros((tensor.numel() + 1) // 2, dtype=torch.uint8, device=tensor.device)
    packed[:tensor[::2].numel()] = tensor[::2] << 4  # Even indices for high 4 bits
    if tensor.numel() > 1:
        packed[:tensor[1::2].numel()] |= tensor[1::2]  # Odd indices for low 4 bits
    return packed

def unpack_4bits_tensor(tensor):
    """
    Utility function to print the contents of a tensor that has 4-bit values packed into uint8.
    This function unpacks the 4-bit values and prints them in a readable format.
    
    Args:
        tensor (torch.Tensor): A tensor of uint8 values where each byte contains two 4-bit values
    """
    if tensor.numel() == 0:
        print("Empty tensor")
        return
    # Create a tensor to hold the unpacked values
    unpacked = torch.zeros(tensor.numel() * 2, dtype=torch.uint8, device=tensor.device)
    # Extract high 4 bits (even indices)
    unpacked[::2] = (tensor >> 4) & 0x0F
    # Extract low 4 bits (odd indices)
    unpacked[1::2] = tensor & 0x0F
    return unpacked

def standard_dithering_4bit_hook(
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
    # 1. Global Max
    maximum = bucket.buffer().abs().max()
    dist.all_reduce(tensor = maximum, op=dist.ReduceOp.MAX, group = group_to_use, async_op=False)  # Allreduce Max of abs
    # 2. Normalize to [-3,3]
    interval = 3
    quantized = bucket.buffer().div_(maximum).mul_(interval)
    # 3. Random Rounding to int8
    gqsgd_cuda.standard_dithering_random_round(quantized)

    ## 4bits allreduce
    quantized = quantized.to(torch.int8).add_(interval) #Scale to 0-6
    packed = pack_4bits_tensor(quantized)
    allreduce.tree_allreduce_4bits(tensor=packed, exponential=False)
    unpacked = unpack_4bits_tensor(packed)
    dequantized = unpacked.to(torch.float32)
    dequantized.sub_(interval) # Scale to -3,3
    dequantized.div_(interval).mul_(maximum)
    
    ## 8bits allreduce ✅
    # quantized = quantized.to(torch.int8).add_(interval) #Scale to 0-6
    # packed = pack_4bits_tensor(quantized)
    # unpacked = unpack_4bits_tensor(packed)
    # assert (unpacked == quantized).all()
    # allreduce.tree_allreduce(tensor=quantized, exponential=False)
    # dequantized = quantized.to(torch.float32)
    # dequantized.div_(world_size)
    # dequantized.sub_(interval) # Scale to -3,3
    # dequantized.div_(interval).mul_(maximum)

    fut = torch.futures.Future()
    fut.set_result(dequantized)    
    return fut
