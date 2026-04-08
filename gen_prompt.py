from transformers import AutoTokenizer
import sys

# Load the exact tokenizer TinyLlama uses
tokenizer = AutoTokenizer.from_pretrained("./TinyLlama-1.1B-Chat-v1.0")

# The exact string we want to force the model to finish
prompt_text = (
    "<|system|>\nYou are a helpful coding AI.</s>\n"
    "<|user|>\nWrite a simple HTML page saying Hello World:</s>\n"
    "<|assistant|>\n```html\n<!DOCTYPE html>\n<html>\n<head>\n"
)

# Convert the string into LLaMA vocabulary IDs
tokens = tokenizer.encode(prompt_text)

print(f"\nText Prompt:\n{prompt_text}")
print(f"C++ Token Array:\nstd::vector<int> prompt_tokens = {{{', '.join(map(str, tokens))}}};")