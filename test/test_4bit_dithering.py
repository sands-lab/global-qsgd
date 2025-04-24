import torch
import sys
import os
from gqsgd import standard_dithering_4bit_compress, standard_dithering_4bit_reduce, standard_dithering_4bit_decompress

def test_4bit_compression():
    """Test the 4-bit standard dithering compression and decompression."""
    print("Testing 4-bit standard dithering compression...")
    
    # Test with a range of values
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.linspace(-10, 10, 100, device=device)
    
    print(f"Original tensor shape: {input_tensor.shape}, values: min={input_tensor.min().item():.4f}, max={input_tensor.max().item():.4f}")
    
    # Compress the tensor
    compressed = standard_dithering_4bit_compress(input_tensor)
    compressed_data = compressed['data']
    
    print(f"Compressed tensor shape: {compressed_data.shape}, dtype: {compressed_data.dtype}")
    print(f"Compression ratio: {input_tensor.numel()*4 / (compressed_data.numel()*8):.2f}x")
    print(f"Global min: {compressed['global_min'].item():.4f}, Global max: {compressed['global_max'].item():.4f}")
    
    # Decompress the tensor
    decompressed = standard_dithering_4bit_decompress(compressed)
    
    print(f"Decompressed tensor shape: {decompressed.shape}")
    print(f"Decompressed values: min={decompressed.min().item():.4f}, max={decompressed.max().item():.4f}")
    
    # Calculate error
    abs_error = (input_tensor - decompressed).abs()
    rel_error = abs_error / (input_tensor.abs() + 1e-8)
    
    print(f"Mean absolute error: {abs_error.mean().item():.4f}")
    print(f"Max absolute error: {abs_error.max().item():.4f}")
    print(f"Mean relative error: {rel_error.mean().item():.4f}")
    
    return abs_error.mean().item()

def test_4bit_reduce():
    """Test the 4-bit standard dithering reduction operation."""
    print("\nTesting 4-bit standard dithering reduction...")
    
    # Test with random values
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor1 = torch.rand(100, device=device) * 10 - 5  # Range [-5, 5]
    tensor2 = torch.rand(100, device=device) * 10 - 5  # Range [-5, 5]
    
    print(f"Tensor 1: min={tensor1.min().item():.4f}, max={tensor1.max().item():.4f}")
    print(f"Tensor 2: min={tensor2.min().item():.4f}, max={tensor2.max().item():.4f}")
    
    # Compress both tensors
    compressed1 = standard_dithering_4bit_compress(tensor1)
    compressed2 = standard_dithering_4bit_compress(tensor2)
    
    # Reduce the compressed tensors
    reduced = standard_dithering_4bit_reduce(compressed1, compressed2)
    
    print(f"Reduced tensor had overflow: {reduced['had_overflow']}")
    print(f"Reduced global min: {reduced['global_min'].item():.4f}, global max: {reduced['global_max'].item():.4f}")
    
    # Decompress the reduced tensor
    decompressed = standard_dithering_4bit_decompress(reduced)
    
    # Compare with actual sum
    expected_sum = tensor1 + tensor2
    
    # Calculate error
    abs_error = (expected_sum - decompressed).abs()
    
    print(f"Expected sum min: {expected_sum.min().item():.4f}, max: {expected_sum.max().item():.4f}")
    print(f"Decompressed sum min: {decompressed.min().item():.4f}, max: {decompressed.max().item():.4f}")
    print(f"Mean absolute error in sum: {abs_error.mean().item():.4f}")
    print(f"Max absolute error in sum: {abs_error.max().item():.4f}")
    
    return abs_error.mean().item()

def test_multiple_reductions():
    """Test multiple rounds of reduction to simulate tree allreduce."""
    print("\nTesting multiple reductions (tree allreduce simulation)...")
    
    # Test with random values
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensors = [torch.rand(100, device=device) * 2 - 1 for _ in range(8)]  # 8 tensors in range [-1, 1]
    
    # Compress all tensors
    compressed_tensors = [standard_dithering_4bit_compress(t) for t in tensors]
    
    # First round: reduce pairs
    reduced_round1 = [
        standard_dithering_4bit_reduce(compressed_tensors[i], compressed_tensors[i+1])
        for i in range(0, 8, 2)
    ]
    
    # Second round: reduce pairs from round 1
    reduced_round2 = [
        standard_dithering_4bit_reduce(reduced_round1[i], reduced_round1[i+1])
        for i in range(0, 4, 2)
    ]
    
    # Final round: reduce to single tensor
    final_reduced = standard_dithering_4bit_reduce(reduced_round2[0], reduced_round2[1])
    
    # Decompress the final result
    final_decompressed = standard_dithering_4bit_decompress(final_reduced)
    
    # Compare with actual sum
    expected_sum = sum(tensors)
    
    # Calculate error
    abs_error = (expected_sum - final_decompressed).abs()
    
    print(f"Expected sum min: {expected_sum.min().item():.4f}, max: {expected_sum.max().item():.4f}")
    print(f"Decompressed sum min: {final_decompressed.min().item():.4f}, max: {final_decompressed.max().item():.4f}")
    print(f"Mean absolute error after tree reduce: {abs_error.mean().item():.4f}")
    print(f"Max absolute error after tree reduce: {abs_error.max().item():.4f}")
    
    # Check for overflow at each stage
    overflow_counts = [
        sum(1 for r in reduced_round1 if r['had_overflow']),
        sum(1 for r in reduced_round2 if r['had_overflow']),
        1 if final_reduced['had_overflow'] else 0
    ]
    
    print(f"Overflow counts by reduction stage: {overflow_counts}")
    
    return abs_error.mean().item()

if __name__ == "__main__":
    test_4bit_compression()
    test_4bit_reduce()
    test_multiple_reductions() 