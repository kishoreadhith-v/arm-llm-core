# Arm-LLM-Core

**Arm-LLM-Core** is a custom LLaMA-architecture inference engine written purely in C++, slightly optimized for ARM processors, built on Apple Silicon (M2). It runs generative AI models locally by focusing on low-level memory management and hardware-specific SIMD acceleration.

![C++](https://img.shields.io/badge/C++-17-blue.svg)
![Build](https://img.shields.io/badge/build-CMake-green.svg)
![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20%7C%20ARM-lightgrey.svg)

## Features & Architecture

This project was built from the ground up to understand the low-level details of Large Language Models. 

### 1. Zero-Copy Memory Management
Instead of reading massive 1GB+ weight files into RAM conventionally, the engine utilizes a **Memory-Mapped (`mmap`)** system. Wrappers like `ModelLoader` pair with lightweight `Tensor` views to map structural metadata directly over data on the disk. RAM is populated instantly via page faults on demand, enabling sub-second load times for large model files. The memory paradigm strictly follows C++ **RAII** (Resource Acquisition Is Initialization) to guarantee leak-free resource release when moving out of scope.

### 2. The Transformer Architecture
The engine implements the core LLaMA mathematical primitives directly:
* **RMSNorm (Root Mean Square Normalization):** For stabilizing the signal between multi-head attention and feed-forward layers.
* **RoPE (Rotary Positional Embeddings):** Applying positional encoding by rotating pairs of coordinates in the query/key projections.
* **Self-Attention & KV Cache:** Employs a pre-allocated Key-Value cache that safely locks context bounds (`max_seq_len`) and calculates Scaled Dot-Product Attention token by token.
* **SiLU & Feed Forward:** Core activations and projection routing in the FFN blocks.
* **Sampling:** Includes robust token sampling capabilities wrapping raw logits with `temperature` adjustments to encourage variability without breaking into infinite/`NaN` spaces.

### 3. Hardware Acceleration (ARM NEON)
The defining feature of this engine is its optimization for Apple Silicon (`-mcpu=apple-m2`) using **ARM NEON SIMD** intrinsics. The core linear operations (Matrix Multiplications inside `math_engine.h`) utilize 128-bit vector registers (`float32x4_t`) via instructions like `vld1q_f32`, fused multiply-accumulates (`vmlaq_f32`), and lane reductions (`vaddvq_f32`) to achieve maximum processing throughput per clock cycle. 

---

## Getting Started

### 1. Build the Engine
The project uses an out-of-source CMake build system. The compiler flags (`-O3`) ensure maximal mathematical and compiler loop unrolling.

```bash
chmod +x build.sh
./build.sh
```

### 2. Export a PyTorch Model
You can convert compatible HuggingFace models (e.g., *TinyLlama-1.1B*) into the raw binary format `.bin` required by the C++ engine using the provided PyTorch script. 

```bash
# Requires transformers and PyTorch
python export.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 tinyllama_1B.bin
```

*(Note: The exporter automatically handles head expansion to fix grouping variations between the model structure and the inference parameters).*

### 3. Run the Inference
Once built, you can execute the engine:
```bash
./build/llm_engine
```

---

## Roadmap

- [x] **Phase 1:** Zero-copy `mmap` tensor loading & matrix primitives.
- [x] **Phase 2:** Transformer primitives (RMSNorm, RoPE, Self-Attention).
- [x] **Phase 3:** Hardware Acceleration (ARM NEON SIMD integration).
- [ ] **Phase 4:** Model Quantization (INT8 weights compression to halve RAM usage).
- [ ] **Phase 5:** Python CLI Wrapper & threading parallelization (`std::thread` implementation for multi-core scaling).
