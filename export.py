import torch
import numpy as np
import struct

print("🚀 Booting PyTorch Exporter...")

# Engine Configuration (Must match your main.cpp exactly)
dim = 288
hidden_dim = 768
num_layers = 6
vocab_size = 32000

# Helper function to write a PyTorch tensor to raw bytes
def serialize(file, tensor):
    # Convert to float32, flatten to 1D, and write to disk
    d = tensor.detach().cpu().to(torch.float32).numpy()
    file.write(d.tobytes())

print("🧠 Generating randomized PyTorch weights...")
# We use torch.randn to simulate a real, untrained neural network
with open("model.bin", "wb") as f:
    
    # 1. Embeddings
    serialize(f, torch.randn(vocab_size, dim))
    
    # 2. Layer Stack
    for i in range(num_layers):
        # Attention Norm
        serialize(f, torch.ones(dim)) # Norm weights usually start at 1.0
        
        # Q, K, V, O
        serialize(f, torch.randn(dim, dim))
        serialize(f, torch.randn(dim, dim))
        serialize(f, torch.randn(dim, dim))
        serialize(f, torch.randn(dim, dim))
        
        # FFN Norm
        serialize(f, torch.ones(dim))
        
        # FFN Gate, Up, Down (Notice the dimensions match your C++ swap!)
        serialize(f, torch.randn(hidden_dim, dim))
        serialize(f, torch.randn(hidden_dim, dim))
        serialize(f, torch.randn(dim, hidden_dim))
        
    # 3. Final Norm
    serialize(f, torch.ones(dim))
    
    # 4. Output Classifier
    serialize(f, torch.randn(vocab_size, dim))

print("✅ Successfully exported raw weights to model.bin!")