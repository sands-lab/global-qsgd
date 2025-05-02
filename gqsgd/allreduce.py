import torch
import torch.distributed as dist
import gqsgd_cuda
import math

""" Implmentation of Parameter Server Allreduce"""
def ps_allreduce(tensor, exponential=False):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = tensor.clone()
    recv_buffs = [tensor.clone() for _ in range(size)]  # Separate buffer for each worker
    reqs = []

    if rank == 0:  # Parameter Server
        # Post all receives first (non-blocking)
        for i in range(1, size):
            reqs.append(dist.irecv(recv_buffs[i], src=i))
        # Process results as they arrive
        for i, req in enumerate(reqs, 1):  # start enumeration from 1
            req.wait()  # Wait for any receive to complete
            if exponential:
                gqsgd_cuda.exponential_dithering_reduce(send_buff[:], recv_buffs[i][:])
            else:
                send_buff[:] += recv_buffs[i][:]

        # Broadcast aggregated results back to all workers
        for i in range(1, size):
            dist.isend(send_buff, dst=i).wait()

    else:  # Workers
        # Send local tensor to parameter server
        dist.isend(send_buff, dst=0).wait()
        # Receive aggregated result from parameter server
        dist.recv(send_buff, src=0)
    
    tensor[:] = send_buff[:]


""" Implementation of a ring-reduce with customized add """
def ring_allreduce(tensor, exponential):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = tensor.clone()
    recv_buff = tensor.clone()
    accum = tensor.clone()
    left = ((rank - 1) + size) % size
    right = (rank + 1) % size
    # Ring AllReduce
    for i in range(size - 1):
        if i % 2 == 0:
            # Send send_buff
            if rank == 0:
                send_req = dist.isend(send_buff, right)
                dist.recv(recv_buff, left)
            else:
                dist.recv(recv_buff, left)
                send_req = dist.isend(send_buff, right)
            tmp=recv_buff[:].clone()
            if exponential == True:
                gqsgd_cuda.exponential_dithering_reduce(accum[:], tmp)
            else:
                accum[:] += tmp

        else:
            # Send recv_buff
            if rank == 0:
                send_req = dist.isend(recv_buff, right)
                dist.recv(send_buff, left)
            else:
                dist.recv(send_buff, left)
                send_req = dist.isend(recv_buff, right)
            tmp=send_buff[:].clone()
            if exponential == True:
                gqsgd_cuda.exponential_dithering_reduce(accum[:], tmp)
            else:
                accum[:] += tmp
        send_req.wait()
    tensor[:] = accum[:]


""" Implementation of a tree-reduce with customized add """
def isSender(rank, layer, interval):
    first_sender = 0
    for i in range(layer):
        first_sender += 2 ** i
    return rank >= first_sender and (rank-first_sender) % (2*interval) == 0

def isReceiver(rank, layer, interval):
    first_sender = 0
    for i in range(layer):
        first_sender += 2 ** i
    return rank >= first_sender and (rank-first_sender) % (2*interval) == interval

def tree_allreduce(tensor, exponential = False):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = tensor.clone()
    recv_buff = tensor.clone()
    layers = int(math.log2(size))
    if(layers != math.log2(size)):
        raise ValueError("The size of the world must be power of 2")
    # Reduce
    for i in range(layers):
        interval = 2 ** (i)
        if isSender(rank, i, interval):
            dist.isend(send_buff, rank + interval, tag = i)
        elif isReceiver(rank, i, interval):
            dist.recv(recv_buff, rank - interval, tag = i)
            if exponential == True:
                gqsgd_cuda.exponential_dithering_reduce(send_buff[:], recv_buff[:])
            else:
                send_buff[:] += recv_buff[:]
    # Broadcast
    # dist.broadcast(send_buff, size-1)
    for i in range(layers-1, -1, -1):
        interval = 2 ** (i)
        if isSender(rank, i, interval):
            dist.recv(send_buff, rank + interval, tag = i+layers)
        elif isReceiver(rank, i, interval):
            dist.isend(send_buff, rank - interval, tag = i+layers)
    tensor[:] = send_buff[:]

def tree_allgather(tensor) -> torch.Tensor:
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = tensor.clone().unsqueeze(0)
    Allgather_tensor = [send_buff.clone() for i in range(size)]
    Allgather_tensor = torch.cat(Allgather_tensor, 0)
    layers = int(math.log2(size))
    if(layers != math.log2(size)):
        raise ValueError("The size of the world must be power of 2")
    # Gather
    for i in range(layers):
        interval = 2 ** (i)
        recv_buff = send_buff.clone()
        if isSender(rank, i, interval):
            dist.isend(send_buff, rank + interval, tag = i)
        elif isReceiver(rank, i, interval):
            dist.recv(recv_buff, rank - interval, tag = i)
            send_buff = torch.cat((recv_buff,send_buff), 0)
    # Broadcast
    # dist.broadcast(send_buff, size-1)
    # print("Rank:",rank, "", send_buff.size())
    for i in range(layers-1, -1, -1):
        interval = 2 ** (i)
        if isSender(rank, i, interval):
            dist.recv(Allgather_tensor, rank + interval, tag = i+layers)
            send_buff = Allgather_tensor
        elif isReceiver(rank, i, interval):
            Allgather_tensor = send_buff
            dist.isend(send_buff, rank - interval, tag = i+layers)
    # print("Rank:",rank, "", Allgather_tensor.size())
    # print(tensor_list)
    return Allgather_tensor
    

def add_packed_4bits(a: torch.Tensor, b: torch.Tensor, level: int = 0, total_levels: int = 1) -> torch.Tensor:
    """
    Add two tensors with 4-bit values packed into uint8 (two 4-bit values per byte).
    Properly handles addition without cross-boundary carry issues.
    
    Args:
        a: Tensor with packed 4-bit values (uint8)
        b: Tensor with packed 4-bit values of the same shape as a (uint8)
        level: Current tree level (0-based)
        total_levels: Total number of tree levels
        
    Returns:
        Tensor with the sum of the 4-bit values, properly packed (uint8)
    """
    # Ensure we're working with uint8
    if a.dtype != torch.uint8:
        a = a.to(torch.uint8)
    if b.dtype != torch.uint8:
        b = b.to(torch.uint8)
        
    # Extract high and low 4 bits from both inputs
    a_high = (a >> 4) & 0x0F
    a_low = a & 0x0F
    b_high = (b >> 4) & 0x0F
    b_low = b & 0x0F
    
    # Add them separately
    high_sum = a_high + b_high
    low_sum = a_low + b_low
    
    # For 4 workers (2 levels), we want to:
    # - Level 0: No division (pairs of workers)
    # - Level 1: Divide by 4 to get the final average across all 4 workers
    if level == total_levels - 1 and total_levels > 1:
        # At the final level, divide by 2^total_levels = world_size to get average
        # For 4 workers, divide by 4
        divisor = 2**total_levels
        
        # Divide high bits with random rounding
        high_remainder = high_sum % divisor
        high_rand = torch.rand_like(high_sum.float()) < (high_remainder.float() / divisor)
        high_bits = high_sum // divisor + high_rand.to(high_sum.dtype)
        
        # Divide low bits with random rounding
        low_remainder = low_sum % divisor
        low_rand = torch.rand_like(low_sum.float()) < (low_remainder.float() / divisor)
        low_bits = low_sum // divisor + low_rand.to(low_sum.dtype)
    else:
        # At earlier levels or single level trees, just clamp to 4 bits
        high_bits = torch.clamp(high_sum, 0, 15)
        low_bits = torch.clamp(low_sum, 0, 15)
    
    # Pack results back into bytes, ensuring we keep within 4 bits for each part
    result = ((high_bits & 0x0F) << 4) | (low_bits & 0x0F)
    
    return result

def tree_allreduce_4bits(tensor, exponential=False):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = tensor.clone()
    recv_buff = tensor.clone()
    layers = int(math.log2(size))
    if(layers != math.log2(size)):
        raise ValueError("The size of the world must be power of 2")
    
    # Reduce
    for i in range(layers):
        interval = 2 ** (i)
        if isSender(rank, i, interval):
            dist.isend(send_buff, rank + interval, tag=i)
        elif isReceiver(rank, i, interval):
            dist.recv(recv_buff, rank - interval, tag=i)
            if exponential == True:
                gqsgd_cuda.exponential_dithering_reduce(send_buff[:], recv_buff[:])
            else:
                # Add packed 4-bit values using the helper function
                # Pass the current level i and total layers to handle scaling
                send_buff[:] = add_packed_4bits(send_buff[:], recv_buff[:], level=i, total_levels=layers)
    
    # Broadcast
    for i in range(layers-1, -1, -1):
        interval = 2 ** (i)
        if isSender(rank, i, interval):
            dist.recv(send_buff, rank + interval, tag=i+layers)
        elif isReceiver(rank, i, interval):
            dist.isend(send_buff, rank - interval, tag=i+layers)
    tensor[:] = send_buff[:]
