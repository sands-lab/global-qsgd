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

def standard_dithering_4bit_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements 4-bit standard dithering quantization.
    It quantizes the gradient tensor to 4-bit unsigned integer (0-7) using standard dithering.
    The quantized tensor will be allreduced then dequantized to the original data type.

    Example::
        >>> ddp_model.register_comm_hook(process_group(None), standard_dithering_4bit_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    INTERVALS = 7  # 4bit -> 0-7, reserve 1 bit for overflow

    # 1. Global Min and Max
    local_min = bucket.buffer().min()
    local_max = bucket.buffer().max()
    global_min = torch.tensor([local_min], device=bucket.buffer().device)
    global_max = torch.tensor([local_max], device=bucket.buffer().device)
    dist.all_reduce(global_min, op=dist.ReduceOp.MIN, group=group_to_use)
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group_to_use)

    # 2. 归一化到[0,1]
    normalized = (bucket.buffer() - global_min) / (global_max - global_min)
    # 3. 量化到[0,INTERVALS]
    scaled = normalized * INTERVALS
    quantized = torch.clamp(scaled.round(), 0, INTERVALS).to(torch.uint8)

    # 4. 打包
    packed = torch.zeros((quantized.numel() + 1) // 2, dtype=torch.uint8, device=quantized.device)
    packed[:quantized[::2].numel()] = quantized[::2] << 4  # 偶数位放高4位
    if quantized.numel() > 1:
        packed[:quantized[1::2].numel()] |= quantized[1::2]  # 奇数位放低4位

    # 5. 通信
    # allreduce.tree_allreduce(tensor=packed, exponential=False)

    # 6. 解包
    unpacked = torch.zeros(quantized.numel(), dtype=torch.uint8, device=packed.device)
    num_even = min(packed.numel(), unpacked.numel() // 2 + unpacked.numel() % 2)
    unpacked[:2*num_even:2] = (packed[:num_even] >> 4) & 0x0F
    num_odd = min(packed.numel(), unpacked.numel() // 2)
    unpacked[1:2*num_odd:2] = packed[:num_odd] & 0x0F

    # 7. 反归一化还原浮点
    dequantized = unpacked.to(torch.float32) / INTERVALS * (global_max - global_min) + global_min

    fut = torch.futures.Future()
    fut.set_result(dequantized)
    return fut