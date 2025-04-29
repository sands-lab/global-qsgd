import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from gqsgd.ddphook import standard_dithering_4bit_hook

# Dummy GradBucket class for testing
class DummyBucket:
    def __init__(self, tensor):
        self._tensor = tensor
    def buffer(self):
        return self._tensor

def run(rank, size):
    # Set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # Create a test tensor with different values on each rank
    # Example: rank 0 has values from -100 to 100, rank 1 has values from -90 to 110, etc.
    input = torch.linspace(-100, 100, 10, dtype=torch.float32, device=device) + rank * 10
    bucket = DummyBucket(input.clone())

    # Initialize accumulated_result before the loop
    accumulated_result = torch.zeros_like(input)
    
    # Run the hook 1000 times and accumulate results
    repeat = 100
    for i in range(repeat):
        # Call the hook (this will do 4-bit quantization, allreduce, and dequantization)
        bucket = DummyBucket(input.clone())  # Reset bucket for each iteration
        fut = standard_dithering_4bit_hook(dist.group.WORLD, bucket)
        result = fut.value()  # This is the dequantized tensor
        accumulated_result += result

    # Calculate average result after 1000 iterations
    average_result = accumulated_result / repeat

    # Compute statistics across all ranks
    all_inputs = [torch.zeros_like(input) for _ in range(size)]
    dist.all_gather(all_inputs, input)
    
    if rank == 0:
        print("Inputs from all ranks:")
        for i, tensor in enumerate(all_inputs):
            print(f"Rank {i}:", tensor)
        
        expected_mean = torch.stack(all_inputs).mean(dim=0)
        print("\nExpected mean after allreduce:", expected_mean)
        print("Actual average result on rank 0:", average_result)
        print("Mean absolute error:", (expected_mean - average_result).abs().mean().item())
        print("Max absolute error:", (expected_mean - average_result).abs().max().item())


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    dist.destroy_process_group()


if __name__ == "__main__":
    size = 4  # Number of processes to spawn
    processes = []
    mp.set_start_method("spawn", force=True)
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()