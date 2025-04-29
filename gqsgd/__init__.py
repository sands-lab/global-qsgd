import gqsgd_cuda
import torch.distributed as dist
from . import ddphook, allreduce, lgreco_hook, powerSGD_hook



__all__=["ddphook","allreduce","lgreco_hook","powerSGD_hook",
         "exponential_dithering_compress", "exponential_dithering_decompress", 
         "exponential_dithering_reduce", "standard_dithering_random_round", "get_world_size"]



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