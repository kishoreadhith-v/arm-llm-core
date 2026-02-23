Notes

This document contains the notes I write myself while learning to build the engine. It's a mix of the math, the cpp and the low-level details.

## The Plan

# Phase 1: The Foundation (Weeks 1-2)

**Goal:** Build the infrastructure.

**Tasks:**

- Setup CMake build system
- Write the Matrix and Tensor memory-mapping classes so we can load a 1GB weight file from your hard drive without crashing your Mac

**Outcome:** A C++ program that can read and parse LLM weights instantly.

---

# Phase 2: The Math (Weeks 3-4)

**Goal:** Implement the Transformer architecture.

**Tasks:**

- Write the C++ code for RMSNorm, RoPE (Rotary Positional Embeddings), and the Self-Attention mechanism

**Outcome:** Your engine can actually process a prompt and generate the next token mathematically, even if it's slow.

---

# Phase 3: The Hardware Acceleration (Weeks 5-6)

**Goal:** Inject the speed.

**Tasks:**

- Bring in the ARM NEON intrinsics and L1/L2 Cache tiling we experimented with
- Add `std::thread` to use all 8 cores of your M2 chip simultaneously

**Outcome:** The engine goes from generating 1 token/sec to 30+ tokens/sec.

---

# Phase 4: Quantization - The Final Boss (Weeks 7-8)

**Goal:** Prove you understand AI compression.

**Tasks:**

- Write an INT8 quantizer
- This shrinks the massive float32 arrays down to 8-bit integers, allowing massive models to fit entirely inside your Mac's fast RAM

**Outcome:** The hallmark of a true AI Systems Engineer.

---

# Phase 5: The Packaging & Launch (Week 9 - Late April)

**Goal:** Make it viral and hireable.

**Tasks:**

- Write the Python CLI wrapper
- Build the README.md with your performance graphs
- Publish it to GitHub and Hacker News

---

## Notes:

### Phase 1 notes:

-

-
