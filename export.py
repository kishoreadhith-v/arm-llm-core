import sys
import torch
from transformers import AutoModelForCausalLM
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python export.py <model_directory> <output_file.bin>")
    sys.exit(1)

model_dir = sys.argv[1]
output_file = sys.argv[2]

# TinyLlama 1.1B Configuration
num_heads = 32
num_kv_heads = 4
head_dim = 2048 // 32
num_layers = 22

def serialize(file, tensor):
    d = tensor.detach().cpu().to(torch.float32).numpy()
    file.write(d.tobytes())

# --- NEW: The RoPE Translation Math ---
def permute_for_rope(w, n_heads):
    # Untangles Hugging Face's split-half RoPE into C++ adjacent-pair RoPE
    dim_out, dim_in = w.shape
    hd = dim_out // n_heads
    w = w.view(n_heads, 2, hd // 2, dim_in)
    w = w.transpose(1, 2).reshape(dim_out, dim_in)
    return w
# --------------------------------------

def repeat_kv(w):
    n_rep = num_heads // num_kv_heads
    w = w.view(num_kv_heads, head_dim, -1)
    w = w.unsqueeze(1).expand(-1, n_rep, -1, -1).reshape(num_heads * head_dim, -1)
    return w

print(f"🧠 Loading weights from {model_dir} into RAM...")
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)
state_dict = model.state_dict()

print(f"📦 Serializing corrected weights to {output_file}...")
with open(output_file, "wb") as f:
    
    # 1. Embeddings
    serialize(f, state_dict["model.embed_tokens.weight"])
    
    # 2. Attention Norms
    for i in range(num_layers):
        serialize(f, state_dict[f"model.layers.{i}.input_layernorm.weight"])
        
    # 3. Wq (Apply RoPE Permute!)
    for i in range(num_layers):
        wq = state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
        serialize(f, permute_for_rope(wq, num_heads))
        
    # 4. Wk (Apply RoPE Permute, THEN repeat for GQA!)
    for i in range(num_layers):
        wk = state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
        wk = permute_for_rope(wk, num_kv_heads)
        serialize(f, repeat_kv(wk))
        
    # 5. Wv (No RoPE here, just repeat for GQA)
    for i in range(num_layers):
        wv = state_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
        serialize(f, repeat_kv(wv))
        
    # 6. Wo 
    for i in range(num_layers):
        serialize(f, state_dict[f"model.layers.{i}.self_attn.o_proj.weight"])
        
    # 7. FFN Norms 
    for i in range(num_layers):
        serialize(f, state_dict[f"model.layers.{i}.post_attention_layernorm.weight"])
        
    # 8. FFN Gate (w1) 
    for i in range(num_layers):
        serialize(f, state_dict[f"model.layers.{i}.mlp.gate_proj.weight"])
        
    # 9. FFN Down (w2) 
    for i in range(num_layers):
        serialize(f, state_dict[f"model.layers.{i}.mlp.down_proj.weight"])
        
    # 10. FFN Up (w3) 
    for i in range(num_layers):
        serialize(f, state_dict[f"model.layers.{i}.mlp.up_proj.weight"])
        
    # 11. Final Norm
    serialize(f, state_dict["model.norm.weight"])
    
    # 12. Output Classifier
    serialize(f, state_dict["lm_head.weight"])

print("✅ Successfully exported structurally sound binary!")