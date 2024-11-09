import torch 
import torch.nn as nn 

class LoRAAdapter(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: float=1.0):
        super(LoRAAdapter, self).__init__()
        self.original_weight = original_layer.weight # Freeze original weights
        self.alpha = alpha 
        
        # Dimenions of original weights
        self.input_dim, self.output_dim = self.original_weight.size()
        
        # Define low-rank metrices A and B 
        self.A = nn.Parameter(torch.randn(self.input_dim, rank)*0.01) # Low rank matrix A
        self.B = nn.Parameter(torch.randn(rank, self.output_dim)*0.01) # low rank matrix B
        
    
    def forward(self, x):
        # compute the low-rank adaptation
        lora_update = torch.matmul(self.A, self.B)
        # combine with the original weight
        adapted_weight = self.original_weight + self.alpha * lora_update
        return torch.matmul(x, adapted_weight.t())
    
    
input_dim, output_dim = 768, 768 # Example dimensions
rank = 4 # Choose a low rank
alpha = 0.1 # Scaling factor 

# Define original layer 

original_layer = nn.Linear(input_dim, output_dim)
# Wrap it with the LoRA adapter 
lora_layer = LoRAAdapter(original_layer, rank, alpha)

# Forward pass with some sample data 
x = torch.randn(1, input_dim)
output = lora_layer(x)
print(f"Output shape: {output.shape}")
        