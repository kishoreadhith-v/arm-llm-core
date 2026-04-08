import struct
from transformers import AutoTokenizer

print("🧠 Loading Hugging Face Ground Truth...")
hf_tokenizer = AutoTokenizer.from_pretrained("./TinyLlama-1.1B-Chat-v1.0")

print("📦 Parsing local tokenizer.bin...")
local_vocab = []
vocab_size = 32000

with open("tokenizer.bin", "rb") as f:
    for i in range(vocab_size):
        # 1. Read Score (4 bytes, float)
        score_bytes = f.read(4)
        if not score_bytes: 
            break
        
        # 2. Read Length (4 bytes, int)
        len_bytes = f.read(4)
        length = struct.unpack('i', len_bytes)[0]
        
        # 3. Read String (length bytes)
        string_bytes = f.read(length)
        
        # Save it, safely decoding the raw bytes
        local_vocab.append(string_bytes.decode('utf-8', errors='replace'))

print("\n=== 🕵️ TOKENIZER PARITY CHECK ===")
# Test beginning, middle, the prompt, and the absolute end
test_ids = [1, 2, 13, 28956, 31999]

for tid in test_ids:
    hf_word = hf_tokenizer.decode([tid])
    local_word = local_vocab[tid]
    
    # We use repr() to safely print invisible characters like newlines
    print(f"ID {tid:<5} | HF: {repr(hf_word):<15} | BIN: {repr(local_word)}")
    
    if tid == 31999 and local_word == "":
        print("\n🚨 OFFSET DRIFT DETECTED: The bin file ended before token 31999!")