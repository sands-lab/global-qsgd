import torch
import torch.distributed as dist
import gqsgd_cuda
import math

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
    

