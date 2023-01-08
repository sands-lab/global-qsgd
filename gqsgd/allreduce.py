import torch
import torch.distributed as dist
import gqsgd_cuda
""" Implementation of a ring-reduce with avg """
def standard_dithering_allreduce(tensor):
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
            accum[:] += recv_buff[:]
        else:
            # Send recv_buff
            if rank == 0:
                send_req = dist.isend(recv_buff, right)
                dist.recv(send_buff, left)
            else:
                dist.recv(send_buff, left)
                send_req = dist.isend(recv_buff, right)
            accum[:] += send_buff[:]
        send_req.wait()
    tensor[:] = accum[:]

""" Implementation of a ring-reduce with customized add """
def exponential_dithering_allreduce(tensor):
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
            gqsgd_cuda.exponential_dithering_reduce(accum[:], tmp)
        else:
            # Send recv_buff
            if rank == 0:
                send_req = dist.isend(recv_buff, right)
                dist.recv(send_buff, left)
            else:
                dist.recv(send_buff, left)
                send_req = dist.isend(recv_buff, right)
            tmp=send_buff[:].clone()
            gqsgd_cuda.exponential_dithering_reduce(accum[:], tmp)
        send_req.wait()
    tensor[:] = accum[:]


""" Implementation of a tree-reduce with customized add """
def exponential_dithering_tree_allreduce(tensor):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = tensor.clone()
    recv_buff = tensor.clone()
    layers = int(math.log(size, 2))
    # Recuesive Reduce
    for i in range(layers): # 0, 1
        if rank % (2 ** (i + 1)) == 0:
            dist.send(send_buff, rank + (2 ** i))
            tmp=recv_buff[:].clone()
            gqsgd_cuda.exponential_dithering_reduce(send_buff[:], tmp)
    tensor[:] = send_buff[:]
    # Broadcast

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
            gqsgd_cuda.exponential_dithering_reduce(accum[:], tmp)
        else:
            # Send recv_buff
            if rank == 0:
                send_req = dist.isend(recv_buff, right)
                dist.recv(send_buff, left)
            else:
                dist.recv(send_buff, left)
                send_req = dist.isend(recv_buff, right)
            tmp=send_buff[:].clone()
            gqsgd_cuda.exponential_dithering_reduce(accum[:], tmp)
        send_req.wait()
    tensor[:] = accum[:]