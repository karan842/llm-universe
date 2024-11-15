"""
4-bit QuantiAtion (4NF):
by scaling and rounding weights to fit within
the 4-bit range [-8, 7].
"""
import torch 

# Define sample tensor(weights)
weights = torch.randn(5, 5)
print(weights.dtype)

# Step 1: Scale to fit withing 4-bit range [-8, 7]
scale_factor = 8.0 / weights.abs().max()
quantized_weights = torch.round(weights * scale_factor)

# Step 2: Clip to 4-bit integer range
quantized_weights = torch.clamp(quantized_weights, -8, 7)


# Step 3: Dequantize to approximate the original scale
dequantized_weights = quantized_weights / scale_factor

print("Original Weights:\n", weights)
print("Quantized Weights (4-bit):\n", quantized_weights)
print("Dequantized Weights:\n", dequantized_weights)


# Step 1: First 4-bit Quantization
scale_factor1 = 8.0 / weights.abs().max()
quantized_weights1 = torch.round(weights * scale_factor1)
quantized_weights1 = torch.clamp(quantized_weights1, -8, 7)

# Step 2: Second 4-bit Quantization on the first quantized result
scale_factor2 = 8.0 / quantized_weights1.abs().max()
quantized_weights2 = torch.round(quantized_weights1 * scale_factor2)
quantized_weights2 = torch.clamp(quantized_weights2, -8, 7)

# Step 3: Dequantize back to approximate the original scale
dequantized_weights_double = quantized_weights2 / (scale_factor1 * scale_factor2)

print("Original Weights:\n", weights)
print("First Quantized Weights (4-bit):\n", quantized_weights1)
print("Second Quantized Weights (Double Quantization):\n", quantized_weights2)
print("Dequantized Weights after Double Quantization:\n", dequantized_weights_double)
