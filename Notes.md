# Notes

This document contains the notes I write myself while learning to build the engine. It's a mix of the math, the cpp and the low-level details.

## The Plan

### Phase 1: Foundation Infrastructure : [[commit](https://github.com/kishoreadhith-v/arm-llm-core/commit/a5297c9d9e5db1bee308534e011afeccd1604ef1)]

**Objective:** Establish the foundational infrastructure for the inference engine.

**Deliverables:**

- Implementation of a CMake-based build system
- Development of Matrix and Tensor memory-mapping abstractions to enable efficient loading of large-scale weight files (>1GB) without exhausting system memory resources

**Expected Outcome:** A C++ application capable of instantaneous reading and parsing of large language model weight parameters.

---

### Phase 2: Mathematical Implementation (Weeks 3-4)

**Objective:** Implement the core Transformer architecture components.

**Deliverables:**

- C++ implementation of RMSNorm (Root Mean Square Normalization)
- Implementation of RoPE (Rotary Positional Embeddings)
- Development of the Self-Attention mechanism

**Expected Outcome:** A functional inference engine capable of processing input prompts and generating subsequent tokens through mathematical computation, albeit without performance optimization.

---

### Phase 3: Hardware Acceleration Integration (Weeks 5-6)

**Objective:** Optimize computational performance through hardware-specific acceleration.

**Deliverables:**

- Integration of ARM NEON SIMD intrinsics
- Implementation of L1/L2 cache tiling strategies
- Parallelization using `std::thread` to leverage multi-core architecture (8 cores, M2 processor)

**Expected Outcome:** Achievement of a 30× performance improvement in token generation throughput (from 1 token/sec to 30+ tokens/sec).

---

### Phase 4: Model Quantization (Weeks 7-8)

**Objective:** Implement model compression through quantization techniques.

**Deliverables:**

- Development of an INT8 quantization module
- Reduction of float32 weight arrays to 8-bit integer representation to optimize memory utilization and enable deployment of larger models within available RAM constraints

**Expected Outcome:** Demonstration of advanced AI systems engineering capabilities through successful model compression implementation.

---

### Phase 5: Documentation and Deployment (Week 9)

**Objective:** Prepare the project for public release.

**Deliverables:**

- Development of a Python CLI wrapper interface
- Comprehensive README documentation including performance benchmarks and visualization

---

## Notes:

### Phase 1 notes:

- System Architecture and Tooling:
  - Out-of-Source Builds (CMake): separation of source code and build artifacts
  - Compiler Directives:
    - `-O3` for maximum loop/math optimization
    - `-mcpu=apple-m2` to specify the target architecture
  - Memory Management:
    - Zero-Copy File Loading (`mmap`)
      - memory map weights from storage to RAM without copying
      - RAM populated via page faults on demand, allowing instant loading of large models
  - C++ Paradigms & Patterns:
    - RAII (Resource Acquisition Is Initialization) for safe memory management
      - Constructors and destructors to manage memory leaks by automatically releasing resources on leaving scope
      - The "Memory View" Pattern: `Tensor` is a the metadata with a pointer to `ModelLoader` which actually hods the memory-mapped data.
