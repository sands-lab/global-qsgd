from . import ddphook, allreduce, lgreco_hook, powerSGD_hook

__all__=["ddphook","allreduce","lgreco_hook","powerSGD_hook",
         "exponential_dithering_compress", "exponential_dithering_decompress", 
         "exponential_dithering_reduce", "standard_dithering_random_round", "get_world_size",
         "standard_dithering_4bit_compress", "standard_dithering_4bit_reduce", "standard_dithering_4bit_decompress"]

import torch
import torch.distributed as dist
import gqsgd_cuda

def get_world_size():
    """Get the world size from PyTorch distributed if initialized, otherwise return 1."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

def exponential_dithering_compress(input, global_norm):
    """
    Compress the input tensor using exponential dithering.
    
    Args:
        input: The input tensor to compress.
        global_norm: The global norm tensor for normalization.
    
    Returns:
        The compressed tensor.
    """
    return gqsgd_cuda.exponential_dithering_compress(input, global_norm)

def exponential_dithering_decompress(input, global_norm, world_size=None):
    """
    Decompress the input tensor using exponential dithering.
    
    Args:
        input: The compressed tensor to decompress.
        global_norm: The global norm tensor for denormalization.
        world_size: The world size for normalization. If None, it will be automatically 
                   detected from torch.distributed.
    
    Returns:
        The decompressed tensor.
    """
    if world_size is None:
        world_size = get_world_size()
    return gqsgd_cuda.exponential_dithering_decompress(input, global_norm, world_size)

def exponential_dithering_reduce(input_a, input_b):
    """
    Reduce two compressed tensors.
    
    Args:
        input_a: The first compressed tensor.
        input_b: The second compressed tensor.
    
    Returns:
        The reduced tensor.
    """
    return gqsgd_cuda.exponential_dithering_reduce(input_a, input_b)

def standard_dithering_random_round(input):
    """
    Apply standard dithering random rounding to the input tensor.
    
    Args:
        input: The input tensor to round.
    
    Returns:
        The rounded tensor.
    """
    return gqsgd_cuda.standard_dithering_random_round(input)

def standard_dithering_4bit_compress(input):
    """
    Compress the input tensor using 4-bit standard dithering.
    
    Args:
        input: The input tensor to compress.
    
    Returns:
        The compressed tensor.
    """
    # Find global min and max for the tensor
    global_min = input.min().view(1)
    global_max = input.max().view(1)
    
    # Move to same device as input
    global_min = global_min.to(input.device)
    global_max = global_max.to(input.device)
    
    # Store original size for later decompression
    original_size = input.numel()
    
    # Compress the tensor
    compressed = gqsgd_cuda.standard_dithering_4bit_compress(input, global_min, global_max)
    
    return {
        'data': compressed,
        'global_min': global_min,
        'global_max': global_max,
        'original_size': original_size
    }

def standard_dithering_4bit_reduce(a, b):
    """
    Reduce two compressed 4-bit tensors.
    
    Args:
        a: First compressed tensor dictionary from standard_dithering_4bit_compress.
        b: Second compressed tensor dictionary from standard_dithering_4bit_compress.
    
    Returns:
        The reduced tensor dictionary.
    """
    # Combine global min and max
    combined_min = torch.min(a['global_min'], b['global_min'])
    combined_max = torch.max(a['global_max'], b['global_max'])
    
    # Ensure original sizes match
    assert a['original_size'] == b['original_size'], "Tensor sizes must match for reduction"
    
    # Perform the reduction
    reduced, overflow = gqsgd_cuda.standard_dithering_4bit_reduce(
        a['data'], b['data'], a['original_size']
    )
    
    # Check if we had significant overflow that might affect accuracy
    has_overflow = overflow.sum().item() > 0
    
    return {
        'data': reduced,
        'global_min': combined_min,
        'global_max': combined_max,
        'original_size': a['original_size'],
        'had_overflow': has_overflow
    }

def standard_dithering_4bit_decompress(compressed):
    """
    Decompress a tensor that was compressed using 4-bit standard dithering.
    
    Args:
        compressed: Dictionary containing compressed data and metadata.
    
    Returns:
        The decompressed tensor.
    """
    return gqsgd_cuda.standard_dithering_4bit_decompress(
        compressed['data'],
        compressed['global_min'],
        compressed['global_max'],
        compressed['original_size']
    )