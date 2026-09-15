---
layout: post
comments: true
title: "FlashAttention Pt 1: Personal Notes"
excerpt: "Personal annotations on an exact tiled-attention handbook: GPU memory traffic, online softmax, FlashAttention-1, and architectural compatibility (MHA/MQA/GQA)."
date: 2026-09-13
mathjax: true
---

---

 _This is Part 1 of a two-part FlashAttention series._

---

**How IO-Aware Attention Makes [Transformers](https://chizkidd.github.io/2026/04/17/transformers/) Faster Without Approximating Attention**

The [handbook](https://drive.google.com/file/d/1CLyK-9Cflcvi3fRl3qAHyzYvwJFjCyVg/view) of reference for this blogpost was inspired by this [tweet](https://x.com/techNmak/status/2098057360908685358). The mechanism, in three words: **Tiling + Online Softmax + Recomputation**. Everything in this handbook is elaboration on that summary. 

> A technical handbook on exact tiled attention: GPU memory traffic, online softmax, forward and backward passes, IO complexity, the evolution from FlashAttention-1 through FlashAttention-4, and current framework behavior.

In this blog post, we're covering the fundamental problem FlashAttention-1 (FA1) tackles, the mathematical tricks utilised, the GPU implementation of these math tricks, and the architectural compatibility ([MHA](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#multi-head-attention)/[MQA](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#multi-query-attention-mqa)/[GQA](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#grouped-query-attention-gqa)) of FlashAttention.

---

## **Table of Contents**

[0. Introduction](#0-introduction)
<!-- - [0.1 Standard Naive Attention Implementation](#0-introduction) -->
- [0.1 How to Read This Handbook](#01-how-to-read-this-handbook)
- [0.1 Notation](#02-notation)

[1. The Fundamental Problem](#1-the-fundamental-problem)
- [1.1 What FlashAttention actually optimizes](#11-what-flashattention-actually-optimizes)
- [1.2 The attention equation is not the implementation](#12-the-attention-equation-is-not-the-implementation)
- [1.3 GPU memory hierarchy and why IO matters](#13-gpu-memory-hierarchy-and-why-io-matters)
- [1.4 Why materializing $S$ and $P$ is expensive](#14-why-materializing-s-and-p-is-expensive)
- [1.5 Dense arithmetic is still quadratic](#15-dense-arithmetic-is-still-quadratic)
- [1.6 Memory-efficient exact attention predates FlashAttention](#16-memory-efficient-exact-attention-predates-flashattention)

[2. The Mathematical Trick](#2-the-mathematical-trick)
- [2.1 Tiling queries, keys, and values](#21-tiling-queries-keys-and-values)
- [2.2 Softmax is the difficult part of streaming](#22-softmax-is-the-difficult-part-of-streaming)
- [2.3 Online softmax from first principles](#23-online-softmax-from-first-principles)
- [2.4 The blockwise merge recurrence](#24-the-blockwise-merge-recurrence)
- [2.5 A complete numerical example](#25-a-complete-numerical-example)

[3. Putting the Mathematics onto the GPU](#3-putting-the-mathematics-onto-the-gpu)
- [3.1 FlashAttention forward pass](#31-flashattention-forward-pass)
- [3.2 Why dense FlashAttention is exact](#32-why-dense-flashattention-is-exact)
- [3.3 IO complexity: what the theorem actually says](#33-io-complexity-what-the-theorem-actually-says)
- [3.4 Memory complexity: linear auxiliary state, not linear compute](#34-memory-complexity-linear-auxiliary-state-not-linear-compute)
- [3.5 Causal masking and tile skipping](#35-causal-masking-and-tile-skipping)
- [3.6 Backward pass: recompute instead of save](#36-backward-pass-recompute-instead-of-save)
- [3.7 Why more FLOPs can still be faster](#37-why-more-flops-can-still-be-faster)

[4. Architectural Compatibility](#4-architectural-compatibility)
- [4.1 MHA, MQA, and GQA compatibility](#41-mha-mqa-and-gqa-compatibility)
- [4.2 Variable lengths, local attention, and dropout](#42-variable-lengths-local-attention-and-dropout)

[5. FlashAttention Evolution](#5-flashattention-evolution)
- [5.1 FlashAttention-2: what changed](#51-flashattention-2-what-changed)
- [5.2 FA2 parallelism across sequence tiles](#52-fa2-parallelism-across-sequence-tiles)
- [5.3 FA2 work partitioning and non-matmul FLOPs](#53-fa2-work-partitioning-and-non-matmul-flops)
- [5.4 FlashAttention-3: the Hopper generation](#54-flashattention-3-the-hopper-generation)
- [5.5 FA3 asynchrony: overlap data movement, GEMM, and softmax](#55-fa3-asynchrony-overlap-data-movement-gemm-and-softmax)
- [5.6 FA3 FP8: performance without pretending precision is free](#56-fa3-fp8-performance-without-pretending-precision-is-free)
- [5.7 FlashAttention-4: the Blackwell generation](#57-flashattention-4-the-blackwell-generation)
- [5.8 FA4 and asymmetric hardware scaling](#58-fa4-and-asymmetric-hardware-scaling)
- [5.9 FA4 implementation and current status](#59-fa4-implementation-and-current-status)
- [5.10 FlashAttention-1 through -4 compared](#510-flashattention-1-through-4-compared)

[6. Using FlashAttention in Frameworks](#6-using-flashattention-in-frameworks)
- [6.1 PyTorch scaled-dot-product attention today](#61-pytorch-scaled-dot-product-attention-today)
- [6.2 Exactness is not bitwise identity](#62-exactness-is-not-bitwise-identity)

[7. FlashAttention vs. Other Techniques](#7-flashattention-vs-other-techniques)
- [7.1 FlashAttention vs. PagedAttention](#71-flashattention-vs-pagedattention)
- [7.2 FlashAttention vs. sparse and linear attention](#72-flashattention-vs-sparse-and-linear-attention)

[8. Training vs. Inference](#8-training-vs-inference)
- [8.1 Training, prefill, and decode are different regimes](#81-training-prefill-and-decode-are-different-regimes)

[9 Practical Engineering](#9-practical-engineering)
- [9.1 Common implementation mistakes](#91-common-implementation-mistakes)
- [9.2 Common misconceptions and the practical mental model](#92-common-misconceptions-and-the-practical-mental-model)


## **Appendix**
- [References](#references)
- [Citation](#citation)

---

## 0. Introduction

### 0.1 How to Read This Handbook

What stands out to me is the number of round trips to HBM in naive standard attention. Every intermediate value $(S$, $P$, $O)$ has to be written out and read back. That is the problem FlashAttention is solving.
The handbook itself frames the subject as easiest to understand when three different questions are kept separate:

> 1. **What mathematical function is being computed?** For dense attention, the target remains ordinary scaled dot-product attention.
>
> 2. **How much arithmetic does that function require?** Dense all-pairs query-key scoring remains quadratic in sequence length.
>
> 3. **How does the implementation move data through the GPU memory hierarchy?** This is where FlashAttention changes the algorithmic execution dramatically.

The lesson I keep coming back to here: FLOP count alone doesn't decide wall-clock speed. An algorithm can do essentially the same math, or even recompute intermediates from scratch, and still finish faster just because it shuttles far less data to and from high-bandwidth memory (HBM).

> **Core distinction.** Dense FlashAttention is an *exact* attention algorithm: it does not replace softmax attention with a low-rank, sparse, kernelized, or approximate formula. "Exact" refers to the mathematical attention computation. Floating-point kernels can still differ by small rounding effects because operations are reordered.

The word *exact* is doing real work here. Exactness is a statement about the mathematical function, not about bitwise reproducibility. The kernel is free to reorder floating-point operations. It is not free to change the function being computed.

### 0.2 Notation

For one attention head, let

$$
Q \in \mathbb{R}^{N_q \times d}, \quad K \in \mathbb{R}^{N_k \times d}, \quad V \in \mathbb{R}^{N_k \times d_v}.
$$

For self-attention, typically $N\_q = N\_k = N$. The scaled score matrix is

$$
S = \frac{QK^T}{\sqrt{d}} + B,
$$

where $B$ represents an optional additive mask or bias, and

$$
P = \mathrm{softmax}_{\mathrm{row}}(S), \quad O = PV.
$$

Throughout, **HBM** refers to large off-chip high-bandwidth GPU memory. **On-chip memory** is a broad teaching term for much smaller, faster storage such as registers and shared memory/**SRAM.** Exact hardware details vary by GPU generation.

The practical difference I keep coming back to:

| HBM | SRAM |
|---|---|
| slow | faster |
| large | smaller |
| off-chip | on-chip |

>**System problem:** Where do all those intermediate values $(S, P, O)$ live while the GPU computes them?

That question, not the arithmetic, is what FlashAttention was built to answer.

## 1. The Fundamental Problem

### 1.1 What FlashAttention Actually Optimizes

Start with the ordinary attention function:

$$
O = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right) V.
$$

A textbook implementation usually boils the attention function down to three big steps:

> Form the score matrix $S$, apply row-wise softmax to get $P$, then multiply by $V$. 

Mathematically, that's perfectly fine. **On a GPU, though, writing a giant intermediate to HBM and reading it back is often far more expensive than the equation lets on.**


FlashAttention's real contribution is an **IO-aware implementation**. It splits the computation into tiles that fit in fast on-chip memory, streams blocks of $K$ and $V$ past them, and keeps just enough row-wise softmax state around to produce the exact output. No full $N \times N$ attention matrix ever needs to land in HBM.

My working definition of **IO-awareness** is simple: 
- _Minimize data movement between the different levels of GPU memory (HBM and SRAM), rather than just trying to reduce the number of mathematical operations (FLOPs)._ 

The reasoning behind it is that the speed bottleneck in modern AI hardware is often not how fast the GPU can compute math, but how fast it can **read and write data**. This is the memory-compute tradeoff.



**Problem:** 

```text
compute --> HBM write --> HBM read --> compute
```

The problem is the number of HBM read-write round trips, and its inefficiency on GPU memory management.

**Tiling:**
$$Q K^T \rightarrow Q_i K_j^T \qquad (\text{for small blocks } Q_i, K_j) $$
 - The score block is 
    1. computed,
    2. softmax-processed,
    3. immediately multiplied by its corresponding $V_j$, and
    4. discared

- Example:
    - For $N = 8192$, instead of an $8192 \times 8192$ intermediate matrix, we can process blocks such as $128 \times 128$. Essentially, we process all the necessary $QK$ interactions in manageable pieces.


> **What changes:** the execution schedule, memory traffic, and stored intermediates.<br>
> **What does not change:** the dense scaled-dot-product attention function being evaluated.


This framing matters, because calling FlashAttention "a faster kind of attention" misses the point. It's better understood as an algorithm and kernel family for **evaluating attention efficiently on accelerators.** A model can use causal masking, RoPE, [MQA](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#multi-query-attention-mqa)/[GQA](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#grouped-query-attention-gqa), or any other attention feature and still run a FlashAttention kernel underneath it.

The original paper draws a line between this and approximate attention methods, which cut arithmetic by changing the math itself. Dense FlashAttention doesn't do that. It keeps the math intact. The block-sparse extension from the same paper is a different story, since skipping blocks literally changes which query-key interactions get computed.


### 1.2 The Attention Equation Is Not the Implementation

The equation does not tell you where tensors live. Let's look at a **standard naive attention implementation**:

1. Calculate $S$, store $S$.
2. Read $S$, calculate $P$, store $P$.
3. Read $P$, calculate $O$.

The problem is the number of HBM round trips. A simple materializing implementation conceptually does:

$$
S \leftarrow QK^T / \sqrt{d}, \quad P \leftarrow \mathrm{softmax}(S), \quad O \leftarrow PV.
$$

Prior to FlashAttention, here is what the standard attention implementation looks like in more detail in terms of memory communication. Load $Q, K, V \in \mathbb{R}^{N \times d}$ in HBM, then:

1. Read $Q, K$ from HBM, compute $S$, write $S$ to HBM.
2. Read $S$ from HBM, compute $P$, write $P$ to HBM.
3. Read $P, V$ by blocks from HBM, compute $O$, write $O$ to HBM.
4. Return $O$.

The GPU spends significant time moving an $N^2$ object through memory. **FlashAttention avoids repeatedly moving huge intermediates:**

1. Calculate small $S$ tile, `softmax` tile, use tile with $V\_j$, discard tile.
2. Calculate next tile, and repeat step 1 until all tiles are completed.

>The **key questions I ask when looking at the attention equation:**
>1. How many operations are required?<br>
>2. What data must move between memory levels to perform those operations?

FlashAttention's move is simple: keep the work close. Blocks of $Q$, $K$, and $V$ sit near the compute units, score tiles get built and consumed on the spot, and only the running row statistics plus the output survive across tiles. Nothing else needs to stick around.

> **Algorithmic lesson.** A computational graph is not a memory schedule. Writing $P = \mathrm{softmax}(QK^T)$ on paper does not require an implementation to store all of $QK^T$ or $P$ in off-chip memory at once.

The pattern isn't unique to attention. **Fused kernels, tiling, recomputation, operator scheduling**; they all do the same thing. Spend a little extra arithmetic to avoid dragging huge intermediates through memory. On modern hardware, that's usually a winning bet. Matmul throughput has raced ahead of the rest of the memory hierarchy, so the math is cheap and the data movement is what hurts. FlashAttention-3 (FA3) and FlashAttention-4 (FA4) take this further, and they're upfront about it: the algorithm bends to the hardware, not the other way around. 

### 1.3 GPU Memory Hierarchy and Why IO Matters

GPUs expose a hierarchy rather than one uniform pool of equally fast memory. The names and capacities vary by architecture, but the mental model is:

<div class="mermaid" style="margin: 2rem 0;">
flowchart BT
    HBM[HBM/device memory]
    SRAM[on-chip shared memory/SRAM]
    Registers[Registers/very local state]
    
    HBM -->|load tiles| SRAM
    SRAM -->|feed compute| Registers
    
    classDef memoryDevice stroke:#818cf8,fill:#eef2ff
    classDef memorySRAM stroke:#2dd4bf,fill:#f0fdfa
    classDef memoryRegisters stroke:#a78bfa,fill:#f5f3ff
    
    class HBM memoryDevice
    class SRAM memorySRAM
    class Registers memoryRegisters
</div>

<small class="text-muted d-block text-center">**Figure 1:** GPU Memory Hierarchy.</small>


FlashAttention-1 (FA1) models the HBM/SRAM asymmetry and explicitly optimizes the number of transfers between them. My breakdown of each GPU memory level in the figure above:

**1. HBM:**
- Relatively large
- Relatively slower to access
- Physically farther from individual compute operations as data must travel to the compute units
- Stores model weights, $Q/K/V$, activations, large tensors

**2. SRAM / shared memory:**
- Much smaller
- Faster
- Cheaper reuse cost

**3. Registers:**
- Even smaller
- More local

**Why tiling helps:** Suppose $Q\_i$ (a query tile/ block) needs to interact with many $K/V$ tiles/blocks prior to writing its output state. Instead of constantly moving $Q\_i$ back and forth, we can keep it close to the compute units while processing:

$$
K\_1, V\_1 \to K\_2, V\_2 \to K\_3, V\_3 \to \cdots
$$

Therefore, one loaded $Q\_i$ can participate in lots of computation. This is called **reuse**. Tiling doesn't only make tensors smaller, it also **increases reuse** while a tile is resident on chip. Conversely, $K/V$ blocks can be streamed through query blocks according to the chosen schedule.

> A kernel becomes **IO-aware when the placement and movement of data are part of the algorithm,** rather than an afterthought left to a sequence of separately launched tensor operations.

This does not mean attention is "always memory bound." FA3 and FA4 exist partly because, as hardware changed, the dominant bottlenecks changed too. The bottleneck depends on:
- Sequence length, $N$
- Head dimension, $d_h$
- dtype
- Mask pattern
- GPU generation (Ampere/Hopper/Blackwell)
- Forward vs. backward pass
- Which kernel is running


### 1.4 Why Materializing $S$ and $P$ Is Expensive

The quadratic intermediate becomes concrete very quickly. Suppose a batch contains one sequence, with 32 attention heads, sequence length $N = 8192$, and a two-byte dtype such as FP16 or BF16. Concretely, my arithmetic:

$$
\begin{aligned}
\text{batch} &= 1 \\
d_h &= 32 \\
N &= 8192 \\
\text{dtype } &= \text{FP1/BF16} \\
\text{tensor shape } &= [1, 32, 8192, 8192] \\
2 & \text{ bytes/element} \\
\end{aligned}
$$

At two bytes per element, the total memory is:
$$
32 \times 8192^2 \times 2 = 4{,}294{,}967{,}296 \text{ bytes} = 4 \text{ GiB}.
$$

One dense tensor with shape $[1, 32, 8192, 8192]$ contains $32 \times 8192^2$ elements. At two bytes per element, that is $4 \text{ GiB}$. That is the size of one full score- or probability-like tensor for this example. _Imagine reading and writing this much data._ 


**Naive pipeline:**

<!-- 
$$
QK^T \to \text{huge } S \to \text{softmax} \to \text{huge } P \to PV
$$ -->


A naive decomposition can produce arrays of this size at several stages. That's not to say every framework keeps $S$ and $P$ alive at once, or that compiler fusion hasn't already cut some of the traffic. The example is there to make the problem concrete: a dense $N^2$ intermediate is big enough that rewriting and rereading it can dominate memory and bandwidth.

<div class="mermaid" style="margin: 2rem 0;">
flowchart LR
    QKT["QKᵀ"] --> S["huge S"]
    S --> Softmax["softmax"]
    Softmax --> P["huge P"]
    P --> PV["PV"]

    classDef boxed fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,font-size:22px
    classDef emphasized fill:none,stroke:none,font-size:26px,font-weight:bold

    class QKT,Softmax,PV boxed
    class S,P emphasized
</div>

<small class="text-muted d-block text-center">**Figure 2:** Naive Attention Implementation Pipeline.</small>


**FlashAttention pipeline:**

<!-- $$
Q\_i K\_j^T \to \text{softmax tile} \to \text{tile} \times V\_j \to \text{discard}
$$ -->

FlashAttention skips all of the naive attention decomposition. Score tiles get formed, softmaxed on chip, used to accumulate their contribution from $V$, then thrown away. The full matrix never touches HBM.

<div class="mermaid" style="margin: 2rem 0;">
flowchart LR
    QK["QᵢKⱼᵀ"] --> Softmax["softmax tile"]
    Softmax --> PV["tile × Vⱼ"]
    PV --> Discard["discard tile"]

    classDef boxed fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,font-size:22px
    classDef plain fill:none,stroke:none,font-size:22px

    class QK,Softmax,PV boxed
    class Discard plain
</div>

<small class="text-muted d-block text-center">**Figure 3:** FlashAttention Implementation Pipeline.</small>


> **Important wording:** FlashAttention removes the need to materialize the full attention matrix as an off-chip intermediate. It does not remove the logical pairwise interactions required by dense attention.

This matters most during training, where naive autograd would want those large intermediates saved for the backward pass.

**Important distinction:** FlashAttention acknowledges that $N^2$ interactions exist and says: *Do not store the entire result of these interactions as a giant intermediate if we can consume each piece immediately.*

### 1.5 Dense Arithmetic Is Still Quadratic

FlashAttention changes the memory schedule, not the math. The function itself is still dense attention. That means forming all query-key scores for $N$ tokens with head dimension $d$ still costs work proportional to $N^2 d$. There's no way around that.

Here's the thing about multiplying probabilities by values: it's another dense pairwise matrix multiplication of the same broad order. FlashAttention reshuffles these operations, but it doesn't skip the dense set of query-key interactions. Those still happen.

So three things must not get conflated:

- **Arithmetic complexity:** dense attention stays at $O(N^2 d)$.
- **Large intermediate storage:** FlashAttention avoids an $O(N^2)$ materialized score/probability tensor in HBM.
- **HBM traffic:** the original paper proves a lower IO cost than standard materializing attention under its two-level memory model.

This is how a method makes much longer sequences practical without making long context "free." Double $N$, and you roughly quadruple the number of dense query-key pairs. FlashAttention just attacks the data-movement and memory-footprint side of that.

> If you actually want to reduce the *number* of query-key pairs, you need a different mathematical structure: sparsity, a local pattern, or a different attention formulation. Those choices change model behavior, and they're conceptually separate from dense FlashAttention.

When you report speedups, always separate asymptotic arithmetic from measured runtime. A kernel can run several times faster at the same $O(N^2 d)$ complexity because the **constant factors, occupancy, fusion, and memory traffic change dramatically.**

Essentially, the key takeaways are summarized below:

| Quality | Dense FlashAttention |
|---|---|
| Arithmetic | $\mathcal{O}(N^2 d)$ |
| Full attention intermediates | Avoid $\mathcal{O}(N^2)$ storage |
| HBM traffic | Reduced substantially |

### 1.6 Memory-Efficient Exact Attention Predates FlashAttention

It would be historically inaccurate to say FlashAttention first discovered that exact attention can avoid quadratic memory.

Earlier work by **Rabe and Staats** demonstrated exact memory-efficient attention with quadratic computation but subquadratic memory. They showed that attention does not require $O(N^2)$ memory with respect to $N$. **Milakov and Gimelshein** also performed earlier work on online softmax that showed a memory-efficient, online recurrent methodology of computing the classical stable softmax normalizer. FlashAttention builds the same kind of running-max/running-normalizer idea into tiled attention, while also accumulating the value-weighted output.
The lineage/evolution from stable softmax to FlashAttention is shown in Figure 5 below:

<!-- $$
\text{Stable softmax} \to \text{Online normalization} \to \text{Exact memory-efficient attention} \to \text{FlashAttention} \to \text{FlashAttention-2} \to \text{FlashAttention-3} \to \text{FlashAttention-4}
$$ -->

<div class="mermaid" style="margin: 2rem 0;">
flowchart TD
    Stable["Stable softmax"] --> Online["Online normalization"]
    Online --> Exact["Exact memory-efficient attention"]
    Exact --> Flash["FlashAttention"]

    classDef boxed fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,font-size:16px
    class Stable,Online,Exact,Flash boxed
</div>

<small class="text-muted d-block text-center">**Figure 4:** Memory-Efficient Exact Attention Lineage.</small>

FlashAttention's key contribution was to bring together the techniques below into a practical high-performance algorithm:

1. Tiling
2. Online softmax
3. Fused computation
4. GPU memory hierarchy awareness
5. IO complexity analysis

>_This distinction matters because "memory-efficient" does not automatically mean "IO-optimized for a particular hardware model."_

---

## 2. The Mathematical Trick

### 2.1 Tiling Queries, Keys, and Values

**Most important.** This is the section where the trick lives. 
 >The score tile $S\_{ij}$ is small enough to be processed near the compute units. Its contribution is folded into running row statistics and an output accumulator, then the tile can be discarded.

- Suppose $Q \in \mathbb{R}^{N\_q \times d}$, $K \in \mathbb{R}^{N\_k \times d}$, and $V \in \mathbb{R}^{N\_k \times d\_v}$. Instead of processing everything at once, divide them into blocks. For example:

    $$
    Q: \boxed{Q\_1} \boxed{Q\_2} \boxed{Q\_3} \quad K, V: \boxed{K\_1, V\_1} \boxed{K\_2, V\_2} \boxed{K\_3, V\_3}
    $$

    For one pair ($Q_i, K_j/V_j$), we process that score tile $S_{ij}$ locally:

    $$
    S\_{ij} \doteq \frac{Q\_i K\_j^T}{\sqrt{d}} + B\_{ij},
    $$

    Then, for each query block:

    1. Calculate score tile
    2. Row-wise softmax and update
    3. Multiply by $V\_j$ and accumulate
    4. Update the running output
    5. Discard the tile 
    6. Move to the next tile while ignoring the need to write a global $S$ or $P$ matrix.

    The softmax used at each tile is

    $$
    \mathrm{softmax}(x\_i) = \frac{e^{x\_i}}{\sum\_j e^{x\_j}}.
    $$

- **Matrix multiplication is easy to tile** because it is fundamentally accumulation:

    $$
    AB = \sum\_j A\_j B\_j,
    $$

    so we can calculate pieces and add them.

- **Softmax is harder** because every element depends on the entire row. If we process the first block, we do not know the eventual denominator. Even worse, numerical stability requires knowing the maximum.


- For kernel executions, we choose tile shapes and loop order based on: **on-chip capacity, head dimension, GPU generation, causal structure, and work partitioning.** FA1 uses tile sizes derived from SRAM capacity $M$.

> Tiling alone is not enough. Matrix multiplication tiles compose naturally because sums can be accumulated. Softmax couples every score in a row through a shared maximum and denominator, so we need a way to merge blocks without seeing the full row at once.

That is the key mathematical trick in the next chapters.

### 2.2 Softmax Is the Difficult Part of Streaming

A numerically stable softmax for one row $x\_1, \ldots, x\_N$ uses

$$
m = \max\_j x\_j, \qquad \ell = \sum\_j e^{x\_j - m}, \qquad p\_j = \frac{e^{x\_j - m}}{\ell}.
$$

The subtraction by $m$ prevents overflow from large positive logits. But it seems to create a streaming problem: how can an early block be normalized if a later block may contain a larger maximum?

The answer is to retain sufficient statistics that can be **rescaled** when the maximum changes. Suppose the running state after earlier elements is $(m\_{\text{old}}, \ell\_{\text{old}})$ and a new block has maximum $m\_b$. Define

$$
m\_{\text{new}} = \max(m\_{\text{old}}, m\_b).
$$

Every contribution accumulated under the old maximum can be converted to the new reference by multiplying it by

$$
\alpha \doteq e^{m\_{\text{old}} - m\_{\text{new}}}.
$$

The new block is evaluated relative to the same $m\_{\text{new}}$.

This is not an approximation. It is the identity

$$
e^{x\_j - m\_{\text{old}}}\, e^{m\_{\text{old}} - m\_{\text{new}}} = e^{x\_j - m\_{\text{new}}}.
$$

**Mathematical insight.** The running maximum is a change of numerical reference point. When that reference changes, previously accumulated exponentials can be rescaled exactly in real arithmetic rather than recomputed from scratch.

The online-normalizer recurrence predates FlashAttention and provides the mathematical basis for streaming stable softmax.

**Worked example.** Start with two extreme values to see why the max subtraction matters:

$$
x = [1000, 999].
$$

Directly computing $e^{1000}$ overflows. Instead, set $m = 1000$, so

$$
x - m = [1000 - 1000, \; 999 - 1000] = [0, -1],
$$

and $e^{0} = 1$, $e^{-1} \approx 0.368$ are perfectly manageable.

Now stream two blocks. Let Block 1 $= [2, 1]$ and Block 2 $= [4, 3]$. After Block 1, $m\_{\text{old}} = 2$. Block 1 is evaluated at $m\_{\text{old}}$ as $[e^{0}, e^{-1}]$. When Block 2 arrives with $m\_b = 4$,

$$
m\_{\text{new}} = \max(2, 4) = 4, \qquad \alpha = e^{2 - 4} = e^{-2}.
$$

Rescaling Block 1 under the new reference:

$$
[\alpha e^{0}, \; \alpha e^{-1}] = [e^{-2}, \; e^{-3}].
$$

Block 2 at the new reference:

$$
[e^{4 - 4}, \; e^{3 - 4}] = [e^{0}, \; e^{-1}].
$$

Putting it all together, with $x = [2, 1, 4, 3]$:

$$
\mathrm{softmax}(x) = \frac{[e^{-2}, \; e^{-3}, \; e^{0}, \; e^{-1}]}{e^{-2} + e^{-3} + e^{0} + e^{-1}}.
$$

The streaming computation for Block 1 is

$$
\ell = \sum\_j e^{x\_j - m} = e^{0} + e^{-1} = 1 + e^{-1},
$$

giving normalized outputs $[e^{0}, e^{-1}] / (1 + e^{-1})$. For Block 2, the running normalizer is

$$
\ell\_{\text{new}} = \alpha\, \ell\_{\text{old}} + \sum\_{j \in \text{Block 2}} e^{x\_j - m\_{\text{new}}} = e^{-2}(1 + e^{-1}) + e^{0} + e^{-1},
$$

and the combined numerator is $[\alpha e^{0}, \alpha e^{-1}, e^{0}, e^{-1}] = [e^{-2}, e^{-3}, e^{0}, e^{-1}]$, matching the full softmax computed in one shot.

### 2.3 Online Softmax from First Principles

Process scalar logits $x\_1, x\_2, \ldots$ one at a time. Initialize

$$
m\_0 = -\infty, \qquad \ell\_0 = 0.
$$

After observing $x\_j$, update

$$
m\_j = \max(m\_{j-1}, x\_j), \qquad \ell\_j = \ell\_{j-1} e^{m\_{j-1} - m\_j} + e^{x\_j - m\_j}.
$$

Milakov and Gimelshein show that this produces the same stable softmax normalizer while requiring fewer passes over the input than the conventional safe-softmax procedure.

For attention, we also need the weighted value sum. Introduce an unnormalized accumulator $a$:

$$
a = \sum\_j e^{x\_j - m} v\_j.
$$

When the maximum changes from $m$ to $m'$, rescale both $\ell$ and $a$ by $e^{m - m'}$. Then add the new exponentials and value contributions under the new reference. At the end,

$$
o = \frac{a}{\ell}.
$$

My working version of the update:

$$
a \doteq \sum\_j e^{x\_j - m} v\_j
$$

When the maximum changes:

$$
a\_{\text{old}} \to a\_{\text{old}}\, e^{m\_{\text{old}} - m\_{\text{new}}}
$$

$$
\ell\_j \to \ell\_{j-1}\, e^{m\_{j-1} - m\_j} + e^{x\_j - m\_j}
$$

Then add the new value contributions. At the end:

$$
o = \frac{a}{\ell}.
$$

> For attention, $x\_j$ is not a fixed input vector stored in advance. Each block of logits is generated on demand from a matrix product $Q K\_j^T / \sqrt{d}$ plus mask/bias terms. The online recurrence lets the kernel consume that block immediately.

The same idea works row by row and block by block, which is what makes a tiled exact softmax-attention forward pass possible.

**Mathematical insight II.** We do not need the entire probability vector. We only need enough information to reconstruct its contribution to the final output.

### 2.4 The Blockwise Merge Recurrence

For one query row, suppose the running state after some key blocks is

$$
(m, \ell, a),
$$

where $m$ is the maximum score seen so far, $\ell$ is the stable softmax denominator under that maximum, and $a \in \mathbb{R}^{d\_v}$ is the unnormalized value accumulator.

For a new score block $s \in \mathbb{R}^b$ with matching values $V\_b \in \mathbb{R}^{b \times d\_v}$, let

$$
m\_b = \max(s), \qquad m' = \max(m, m\_b),
$$

$$
\alpha = e^{m - m'}, \qquad p = e^{s - m'}.
$$

Then update

$$
\ell' = \alpha \ell + \sum\_j p\_j,
$$

$$
a' = \alpha a + p^T V\_b,
$$

$$
m \leftarrow m', \qquad \ell \leftarrow \ell', \qquad a \leftarrow a'.
$$

Finally,

$$
o = a / \ell.
$$

The mean of the running state:

- $m$ = max score seen so far
- $\ell$ = stable softmax denominator, exponential sum (normalizer)
- $a$ = unnormalized value accumulator

The blockwise recurrence in my own notation:

- For one query row, maintain $(m, \ell, a)$.
- Suppose the next score block is $s = [s\_1, s\_2, \ldots, s\_b]$ with corresponding values $V\_b \in \mathbb{R}^{b \times d\_v}$.
- First calculate the block maximum: $m\_b = \max(s)$, then update the global maximum: $m' = \max(m, m\_b)$.
- Define: $\alpha = e^{m - m'}$, $p = e^{s - m'}$.
- Then update: $\ell' = \alpha \ell + \sum\_j p\_j$, $a' = \alpha a + p^T V\_b$.
- Then set $m \leftarrow m'$, $\ell \leftarrow \ell'$, $a \leftarrow a'$.
- Finally at the end: $o = a / \ell$.

> For a block of query rows, $m$ and $\ell$ become row-wise vectors and $a$ becomes a matrix. Masks can be applied to the score tile before the exponentials, with masked positions contributing zero probability.

> The recurrence is the algebraic reason tile boundaries do not change the dense softmax result. A different tiling changes the order of floating-point operations, but not the intended real-arithmetic function.

FlashAttention's published algorithms express equivalent running-max / running-normalizer / output updates in block form.

**This recurrence is the heart of tiled exact attention.**

### 2.5 A Complete Numerical Example

Consider one already-scaled, unmasked attention row

$$
s = [2, 1, 4, 3]
$$

with two-dimensional values

$$
v\_1 = [1, 0], \quad v\_2 = [0, 1], \quad v\_3 = [2, 0], \quad v\_4 = [0, 2].
$$

The global maximum is 4, so stable unnormalized weights are

$$
[e^{-2}, e^{-3}, e^{-1}, e^{-1}] \approx [0.135335, 0.049787, 1, 0.367879].
$$

Their sum is

$$
\ell \approx 1.55300179,
$$

and full softmax gives

$$
O \approx [1.37497284, 0.50582424].
$$

Now process two blocks. For $[2, 1]$:

$$
m\_1 = 2, \qquad \ell\_1 = 1 + e^{-1} = 1.36787944, \qquad a\_1 = [1, e^{-1}].
$$

For the second block $[4, 3]$, the new maximum is 4, so

$$
\alpha = e^{2 - 4} = e^{-2}.
$$

Then

$$
\ell\_2 = \alpha \ell\_1 + 1 + e^{-1} = 1.55300179,
$$

$$
a\_2 = \alpha a\_1 + [2, 2 e^{-1}] \approx [2.13533528, 0.78554595].
$$

Therefore

$$
a\_2 / \ell\_2 \approx [1.37497284, 0.50582424],
$$

matching the full-row computation.

> The old block was not revisited. Its contribution was merely rescaled when a larger maximum appeared.

**Tying everything together.** Block 1 was never recomputed when block 2 revealed a larger maximum; we rescaled the statistics from block 1. This is what makes streaming possible.

My step-by-step computation:

The score row: $s = [2, 1, 4, 3]$. Values: $v\_1 = [1, 0]$, $v\_2 = [0, 1]$, $v\_3 = [2, 0]$, $v\_4 = [0, 2]$.

**Blocks:** $[2, 1]$ and $[4, 3]$.

**Block 1.** Scores: $[2, 1]$. Maximum: $m\_1 = 2$. Stable exponentials:

$$
[e^{0}, e^{-1}] = [1, e^{-1}]
$$

So

$$
\ell\_1 = 1 + e^{-1} \approx 1.36788.
$$

Value accumulator:

$$
a\_1 = 1 \cdot [1, 0] + e^{-1} \cdot [0, 1] = [1, e^{-1}].
$$

**Block 2.** Scores: $[4, 3]$. $m\_b = 4$. $m' = \max(2, 4) = 4$. $\alpha = e^{m - m'} = e^{2 - 4} = e^{-2}$.

$$
\ell\_2 = \alpha \ell\_1 + e^{0} + e^{-1} = e^{-2}(1 + e^{-1}) + 1 + e^{-1} = e^{-2} + e^{-3} + 1 + e^{-1} \approx 1.553
$$

$$
a\_2 = e^{-2} \cdot [1, e^{-1}] + [2 e^{0}, 2 e^{-1}] = [2 + e^{-2}, 2 e^{-1} + e^{-3}] \approx [2.135, 0.7855]
$$

**Output.**

$$
O\_2 = a\_2 / \ell\_2 \approx [1.375, 0.506]
$$

---

## 3. Putting the Mathematics onto the GPU

### 3.1 FlashAttention Forward Pass

A useful conceptual forward pass is:

1. Partition $Q$ into query-row tiles and $K, V$ into key/value tiles.
2. For each query tile, initialize row-wise running maxima, normalizers, and output accumulators.
3. Load a key/value tile and form the local score tile $Q\_i K\_j^T / \sqrt{d}$.
4. Apply causal/local masks or additive biases that belong to this tile.
5. Compute the tile maximum, update the running maximum, and rescale previous state.
6. Exponentiate the current tile relative to the updated maximum.
7. Update the denominator and the value-weighted output accumulator.
8. Continue until every required key tile has been contributed.
9. Normalize the accumulator row-wise and write the output.

The original FlashAttention algorithm chooses block sizes so the relevant tiles and state fit in on-chip SRAM, reducing trips to HBM.

An educational implementation can reproduce the algebra in a few lines of PyTorch, but such code is not a high-performance FlashAttention kernel. Production implementations depend on GPU-specific tiling, thread/warp scheduling, asynchronous copies, tensor-core instructions, and other low-level details.

> The essential algorithmic idea is independent of one CUDA kernel: generate a score tile, consume it immediately through online softmax and $V$ accumulation, and never materialize the complete score/probability matrix in HBM.

**Minimal educational PyTorch.**

```python
def tiled_attention(q, k, v, block=128):
    scale = 1 / math.sqrt(q.shape[-1])
    n_q, n_kv = q.shape[0], k.shape[0]
    O = torch.zeros_like(q)

    for i in range(0, n_q, block):
        q_blk = q[i:i+block]
        m = torch.full((q_blk.shape[0],), -float('inf'))
        l = torch.zeros((q_blk.shape[0],))
        a = torch.zeros((q_blk.shape[0], v.shape[-1]))

        for j in range(0, n_kv, block):
            k_blk = k[j:j+block]
            v_blk = v[j:j+block]
            s = (q_blk @ k_blk.T) * scale
            m_new = torch.maximum(m, s.max(dim=-1).values)
            p = torch.exp(s - m_new[:, None])
            alpha = torch.exp(m - m_new)
            l = alpha * l + p.sum(dim=-1)
            a = alpha[:, None] * a + p @ v_blk
            m = m_new

        O[i:i+block] = a / l[:, None]

    return O
```

Conceptually, for each KV block:

- scores = $Q \times K\_{block}^T$
- update running softmax, $\ell$
- update output accumulator, $a$

This isn't a high-performance FlashAttention kernel. Real implementations additionally exploit:

- GPU-specific tiling
- Registers
- Shared memory
- Warp scheduling
- Tensor cores
- Async copies
- Specialized intrinsics
- Occupancy optimization

This distinction will become very important for FA-2/3/4.

Later FlashAttention generations keep this semantic structure while changing how work is scheduled on newer hardware.

My step list for the forward pass:

1. Load $Q\_i$.
2. Initialize $(m, \ell, a) = (-\infty, 0, 0)$ — $m = -\infty$, $\ell = 0$, $a = 0$.
3. Load a $K\_j, V\_j$ tile.
4. Compute the local score tile $S\_{ij} = Q\_i K\_j^T / \sqrt{d}$.
5. Apply masks/biases $B\_{ij} / M\_{ij}$.
6. Find the tile maximum and update the running maximum $\max(m, m\_b)$.
7. Compute exponentials relative to the new maximum.
8. Update $\ell$ and $a$.
9. Move to the next $K, V$ tile until every tile is completed/covered.
10. Normalize the accumulator row-wise $O = a / \ell$, then write $O$.

### 3.2 Why Dense FlashAttention Is Exact

Does FlashAttention approximate attention? For dense FlashAttention, **no**. The target remains

$$
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right) V.
$$

My checklist for why this is exact:

- No Q-K pairs are intentionally removed.
- No low-rank approximation is introduced.
- No alternative kernelized attention function replaces softmax.
- This algorithm simply processes the same interactions in blocks.

But *exact* has a qualification. Let me illustrate.

Suppose two implementations compute $a + b + c$:

- **A:** $(a + b) + c$
- **B:** $a + (b + c)$

In exact mathematics,

$$
(a + b) + c = a + (b + c).
$$

But in floating points, in some cases,

$$
(a + b) + c \neq a + (b + c).
$$

Tiling changes the reduction order. Fusion can change the rounding behavior. Therefore:

$$
\text{exact algorithm} \neq \text{bitwise identical output}.
$$

This distinction becomes important in numerical testing.

> Dense FlashAttention is called exact because the target function remains
>
> $$
> \mathrm{softmax}(QK^T / \sqrt{d} + B) V.
> $$
>
> Tiling does not delete query-key pairs. Online softmax does not replace the exponential or normalization with another function. The block recurrence simply changes the order in which sufficient statistics are accumulated.

Three caveats keep the word *exact* precise:

> - **Floating point is finite precision.** Reordering additions and reductions can produce small numerical differences from another implementation. PyTorch explicitly warns that SDPA backends can differ because floating-point operations are fused and ordered differently.
> - **Dropout is stochastic during training.** Comparing two runs bit-for-bit requires matching random behavior in addition to mathematical attention semantics.
> - **Sparse variants are different.** The original FlashAttention paper also presents block-sparse FlashAttention, which omits blocks and is therefore an approximate/sparse variant relative to full dense attention.

> "Exact" does not mean "bitwise identical to every reference kernel." It means the algorithm is not intentionally changing dense softmax attention to reduce the mathematical work.

This distinction matters when evaluating numerical tests. A sensible tolerance depends on dtype, accumulation order, sequence length, and backend rather than requiring binary identity.

### 3.3 IO Complexity: What the Theorem Actually Says

Let me get more theoretical.

The original FlashAttention paper analyzes a two-level memory model with HBM and on-chip SRAM of size $M$. Under the paper's assumptions, including head dimension $d$ and

$$
d \leq M \leq Nd,
$$

standard materializing attention requires approximately

$$
\Theta(Nd + N^2)
$$

HBM accesses, whereas FlashAttention requires

$$
\Theta\left(\frac{N^2 d^2}{M}\right)
$$

HBM accesses under the specified regime.

The exact theorem has assumptions, so do not interpret this as "FlashAttention always moves exactly this many bytes." It is an asymptotic result for a particular memory model that is **communication-complexity bound.**

**Why does larger SRAM help?** Suppose you have more on-chip memory. You can fit larger tiles. Larger tiles mean:

- more data stays resident
- more reuse
- fewer HBM reloads

So increasing $M$ can decrease the amount of HBM traffic. This is one reason GPU architecture matters so much to FlashAttention performance.

> The original FlashAttention paper analyzes a two-level memory model with HBM and on-chip SRAM of size $M$. Under the paper's assumptions, including head dimension $d$ and $d \leq M \leq Nd$, standard materializing attention requires $\Theta(Nd + N^2)$ HBM accesses, whereas FlashAttention requires $\Theta(N^2 d^2 / M)$ HBM accesses. The paper also proves an optimality result over a range of SRAM sizes in this model.

Several details matter:

> - These are IO-complexity results in a particular memory model, not a universal byte count for every GPU.
> - The quantity counts movement of scalar elements/words between the modeled memory levels, up to asymptotic factors.
> - It is not the arithmetic complexity. Dense attention still performs $O(N^2 d)$ work.
> - Increasing usable on-chip memory $M$ enables more reuse and reduces modeled HBM traffic.

> A common incorrect summary is "FlashAttention reduces attention IO from $O(N^2)$ to $O(N)$." The linear quantity is the large auxiliary memory footprint with respect to sequence length, not the general HBM-access expression above.

Intuitively, more usable on-chip memory lets larger working tiles stay resident and be reused for more attention work before data must be reloaded from HBM. In the theorem's regime, that increased reuse is why the HBM-access bound decreases as $M$ grows.

### 3.4 Memory Complexity: Linear Auxiliary State, Not Linear Compute

Another important distinction: linear auxiliary state is not linear compute.

A naive attention implementation may create $\mathcal{O}(N^2)$ attention intermediates. FlashAttention does not. Instead it maintains:

- $Q/K/V$ tiles
- row-wise $m$
- row-wise $\ell$
- output accumulator $a$

The auxiliary attention memory is

$$
\mathcal{O}(Nd).
$$

For a fixed head dimension $d$, the auxiliary attention state scales linearly with sequence length $N$, but linear memory is not linear computation. The computation is still

$$
\mathcal{O}(N^2 d).
$$

**Example.** For $N = 10{,}000$, there are approximately

$$
10{,}000^2 = 100{,}000{,}000
$$

Q-K interactions. FlashAttention does not make those disappear. However, it prevents you from needing a gigantic intermediate containing all of them.

> Why do FlashAttention papers often say memory becomes linear instead of quadratic?
>
> The inputs and output already contain $O(Nd)$ elements. A materializing dense attention implementation additionally creates $O(N^2)$ score/probability state. FlashAttention avoids storing those full matrices and retains only tiled working state plus row-wise statistics. As a result, the extra memory associated with the attention operation scales linearly with sequence length rather than quadratically.
>
> For training, the difference is especially important because a straightforward backward pass might otherwise save a full probability matrix $P$. FlashAttention instead saves compact information such as output and row-wise normalization statistics, then recomputes score/probability tiles in backward.

> **Linear memory does not imply linear runtime.** The number of dense query-key interactions is still quadratic in $N$.

Also avoid claiming that total model memory is $O(N)$. Other Transformer components consume activation memory, and autoregressive serving has KV-cache memory that grows with context length. FlashAttention is specifically changing how the attention computation manages its intermediates.

One reason FlashAttention and activation checkpointing can coexist: both trade recomputation for reduced stored state, but at different scopes of the training graph.

### 3.5 Causal Masking and Tile Skipping

Consider autoregressive attention. Query token $i$ cannot attend to future key token $j > i$.

The attention matrix looks like

$$
\begin{bmatrix}
\checkmark & \times & \times \\
\checkmark & \checkmark & \times \\
\checkmark & \checkmark & \checkmark
\end{bmatrix}
$$

with mask

$$
M = \begin{bmatrix}
0 & -\infty & -\infty \\
0 & 0 & -\infty \\
0 & 0 & 0
\end{bmatrix}.
$$

Applying the mask before softmax gives

$$
\mathrm{softmax}(S') = \begin{bmatrix}
\text{value} & 0 & 0 \\
\text{value} & \text{value} & 0 \\
\text{value} & \text{value} & \text{value}
\end{bmatrix}.
$$

With tiled attention, we can skip entire tiles:

1. Entire blocks above the diagonal can be entirely skipped.
2. Blocks below the diagonal are fully valid.
3. Diagonal blocks require element-level masking.

This is important because we are no longer doing useless work for obviously invalid future positions.

> In causal self-attention, query position $i$ may not attend to future key positions $j > i$. A materialized mask would be another $N \times N$ object, but an efficient tiled kernel can reason about the geometry of each tile.
>
> For square self-attention:
>
> - blocks strictly above the causal boundary are fully masked and need not contribute,
> - blocks strictly below the boundary are fully valid,
> - only blocks intersecting the diagonal require element-level causal masking.

This avoids computing many invalid tiles in a causal kernel. FlashAttention-2 explicitly exploits causal structure, while current implementations also define precise alignment rules for unequal query/key lengths.

> Mask semantics are an API detail that must not be guessed. For example, current PyTorch SDPA treats a Boolean `attn_mask` value of True as a position that *participates* in attention, while other PyTorch mask APIs use different conventions.

**Connection to local attention.** Suppose each token can only attend to 128 nearby tokens. Then many tiles can also be skipped. But now we have changed the mathematical attention pattern: that is no longer undistributed dense attention. FlashAttention can be the kernel executing the local pattern, but the model itself is now doing sparse/local attention.

> Local/sliding-window attention can similarly skip tiles outside the permitted window. But once the model intentionally restricts which pairs are attended, the *model's attention pattern* is sparse/local. FlashAttention can be the kernel used to execute that pattern, but it is no longer the same mathematical problem as unrestricted dense attention.

### 3.6 Backward Pass: Recompute Instead of Save

This is vital for training.

During the forward pass, we do not save the entire $P$ matrix ($N \times N$). We recompute it.

**Forward:**

1. Store compact info such as:
   - output
   - row-wise normalization statistics
2. Do not store $P \in \mathbb{R}^{N \times N}$.

**Backward:**

1. For each tile:
   - a) Recompute $QK^T$
   - b) Reconstruct the local probability values
   - c) Calculate gradients
   - d) Discard the tile
2. This trades more computation for less memory traffic/storage.

**Conceptually:**
- Forward: don't store $P$.
- Backward: recompute $P$ tile-by-tile.

The gradient rehashing for $P$:

$$
dV \mathrel{+}= P^T dO
$$

$$
dP = dO \cdot V^T
$$

$$
dS = P \odot (dP - D\_i[:, \text{None}])
$$

Where $D\_i[:, \text{None}]$ is the column/vector broadcast across each row.


and then:

$$
dQ \mathrel{+}= dS \cdot K / \sqrt{d}
$$

$$
dK \mathrel{+}= dS^T \cdot Q / \sqrt{d}
$$

with appropriate masking. 

> A naive training implementation can save $P = \mathrm{softmax}(S)$ for backward. FlashAttention avoids keeping that $N^2$ tensor in HBM. Instead, it stores compact row-wise normalization information and recomputes score/probability tiles when gradients are needed.
>
> For
>
> $$
> S = QK^T / \sqrt{d}, \quad P = \mathrm{softmax}(S), \quad O = PV,
> $$
>
> a useful row-wise identity is
>
> $$
> D\_i = \sum\_r (dO\_i)\, O\_{ir} = \sum\_j P\_{ij}\, dP\_{ij}.
> $$
>
> After recomputing a tile of $P$, the local derivatives can be expressed as
>
> $$
> dV \mathrel{+}= P^T dO, \quad dP = dO\, V^T, \quad dS = P \odot (dP - D\_i[\text{:}, \text{None}]),
> $$
>
> then
>
> $$
> dQ \mathrel{+}= dS\, K / \sqrt{d}, \quad dK \mathrel{+}= dS^T\, Q / \sqrt{d}.
> $$
>
> Masks imply zero probability/gradient contribution for masked entries. FA2 stores a row-wise log-sum-exp quantity that allows the probability tile to be reconstructed stably from recomputed scores.

> Backward recomputation is intentional. It spends extra matrix-multiply work to avoid reading and writing a giant probability tensor, which can be a favorable trade on GPUs.

This is the **memory-compute tradeoff** applied to the backward pass.

### 3.7 Why More FLOPs Can Still Be Faster

This is one of the biggest lessons from FlashAttention.

Normally we think: fewer FLOPs $\Rightarrow$ faster. But on GPUs, that's incomplete.

Different operations have radically different throughput. Tensor cores are exceptionally good at matrix multiplication. Other operations have different performance profiles. These operations include:

1. exponentials
2. reductions
3. synchronization
4. shared-memory operations
5. memory transfers

Therefore, sometimes doing **extra arithmetic** is worthwhile if it eliminates expensive memory traffic.

For example, consider two options:

- **A:** Compute $\to$ write huge $P$ to HBM $\to$ read $P$ $\to$ compute.
- **B:** Compute $\to$ discard $\to$ recompute later.

Option B performs more arithmetic. But it might be faster because it avoids moving a giant tensor through HBM.

This is a fundamental ML systems principle:

> The cost of a FLOP depends on what hardware executes it and what data movement surrounds it.

> It is tempting to assume that fewer arithmetic operations always imply lower latency. Accelerator performance breaks that intuition regularly.
>
> Matrix multiplication maps exceptionally well to tensor cores. HBM traffic, synchronization, shared-memory traffic, exponentials, reductions, and kernel launch boundaries can be comparatively expensive.
>
> FA2 makes this contrast explicit: one of its goals is to reduce non-matmul FLOPs, because those operations do not enjoy the same throughput as tensor-core GEMMs. The paper also improves work partitioning so more of the GPU is occupied.
>
> FA3 goes further on Hopper by overlapping matrix multiplication, softmax, and data movement using asynchronous hardware features. FA4 responds to Blackwell, where tensor-core throughput increased faster than some other resources, making exponentials and shared-memory traffic relatively more important.

> A better performance question is not merely "How many FLOPs?" but "Which operations, on which units, with what data movement, reuse, parallelism, and synchronization?"

This is the systems lesson that makes FlashAttention important beyond attention itself: hardware efficiency often comes from co-designing mathematical scheduling with the memory/execution hierarchy.

---

## 4. Architectural Compatibility

### 4.1 MHA, MQA, and GQA Compatibility (Shared K/V Heads)

FlashAttention is not tied to standard MHA.

Let

$$
Q \in \mathbb{R}^{B \times N\_q \times H\_q \times d}, \quad K, V \in \mathbb{R}^{B \times N\_k \times H\_{kv} \times d}.
$$

Three head-sharing regimes:

- **MHA:** $H\_q = H\_{kv}$ $\Rightarrow$ every query head has its own K/V head.
- **MQA:** $H\_{kv} = 1$ $\Rightarrow$ all query heads share 1 K/V head.
- **GQA:** $1 < H\_{kv} < H\_q$ $\Rightarrow$ several query heads share K/V heads.

Vital distinction:

1. MHA/MQA/GQA defines the architecture and head sharing.
2. FlashAttention defines efficient execution of the attention computation.

Thus MHA/MQA/GQA and FlashAttention can coexist. Current implementations impose shape constraints: the number of query heads must be divisible by the number of KV heads:

$$
H\_q \bmod H\_{kv} = 0.
$$

> FlashAttention does not require every architecture to have the same number of query and key/value heads.
>
> For ordinary MHA, $H\_q = H\_{kv}$. In MQA, $H\_{kv} = 1$. In GQA, $1 < H\_{kv} < H\_q$. Current Dao-AILab kernels support MQA/GQA by passing fewer KV heads than query heads, with the requirement that the number of query heads be divisible by the number of KV heads.
>
> The kernel still evaluates attention between each query head and its assigned KV head. Head sharing changes the architecture and KV-memory footprint. FlashAttention changes how the resulting attention operation is executed.

**Orthogonal concepts:**

> - MQA/GQA: how heads share K/V projections,
> - RoPE: how position transforms Q/K,
> - FlashAttention: how attention is scheduled and computed efficiently.
>
> These can be used together.

> Current PyTorch SDPA also exposes `enable_gqa`. Its documentation labels GQA support experimental and imposes backend- and tensor-shape constraints, so production code should follow the exact version's documentation rather than assuming universal fused-kernel support.

### 4.2 Variable Lengths, Local Attention, and Dropout

Real systems aren't always: same sequence length + dense attention + no dropout.

Production implementations may support:

- variable-length sequences
- causal attention
- sliding-window attention (SWA)
- dropout
- MQA/GQA
- ALiBi-style bias
- KV-cache decoding (including optional RoPE handling)

But an important distinction: these are implementation capabilities, not fundamental properties of the FlashAttention mathematical idea.

Feature support depends on:

- GPU
- CUDA/ROCm backend
- dtype
- head dimension
- mask
- library version
- kernel generation

PyTorch issue: `scaled_dot_product_attention` applies dropout according to the supplied `dropout_p`, so eval code should explicitly use `0.0` when dropout should be disabled.

> Production attention rarely consists only of equal-length dense sequences with no dropout. Current FlashAttention implementations support a broader feature set, but these are implementation capabilities, not properties of the mathematical idea itself.
>
> The current Dao-AILab repository documents kernels/interfaces for features including:
>
> - variable-length sequences,
> - causal attention,
> - local/sliding-window attention,
> - dropout in training-oriented interfaces,
> - MQA/GQA,
> - ALiBi-style score bias in relevant interfaces,
> - specialized incremental-decoding paths with KV cache, including optional RoPE handling.

> Feature support differs by CUDA/ROCm backend and evolves over time. For example, the repository documents separate NVIDIA and AMD backends with different implementation details and support matrices.

> Dropout deserves a practical warning. Current PyTorch `scaled_dot_product_attention` always applies dropout according to its `dropout_p` argument, so callers must pass `0.0` during evaluation when dropout should be disabled.

> Do not infer feature support from the name "FlashAttention." Check the exact library, kernel generation, device, dtype, head dimension, mask/bias, and training/inference path that will actually run.

## Appendix: Source Section Mapping

<!-- | Source § | Hierarchical |
|---|---|
| 1-6 | 1.1-1.6 |
| 7-11 | 2.1-2.5 |
| 12-18 | 3.1-3.7 |
| 19-20 | 4.1-4.2 |
| 21-30 | 5.1-5.10 |
| 31-32 | 6.1-6.2 |
| 33-34 | 7.1-7.2 |
| 35 | 8.1 |
| 36-37 | 9.1-9.2 | -->

### **References**

[^1]: Ashish Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.

[^2]: Maxim Milakov and Natalia Gimelshein. [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867). arXiv, 2018.

[^3]: Markus N. Rabe and Charles Staats. [Self-attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682). arXiv, 2021.

[^4]: Tri Dao et al. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). NeurIPS 2022.

[^5]: Tri Dao. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). ICLR 2024.

[^6]: Jay Shah et al. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608). NeurIPS 2024.

[^7]: Ted Zadouri et al. [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451). arXiv, 2026.

[^8]: Dao-AILab. [flash-attention official repository and current implementation documentation](https://github.com/Dao-AILab/flash-attention). GitHub. Accessed September 2026.

[^9]: PyPI. [flash-attn-4 package metadata and release history](https://pypi.org/project/flash-attn-4/). PyPI. Accessed September 2026.

[^10]: PyTorch. [torch.nn.functional.scaled_dot_product_attention documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). PyTorch Documentation. Accessed September 2026.

[^11]: PyTorch. [torch.nn.attention.sdpa_kernel documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html). PyTorch Documentation. Accessed September 2026.

[^12]: PyTorch. [torch.nn.attention module and FlashAttention implementation registration APIs](https://docs.pytorch.org/docs/stable/nn.attention.html). PyTorch Documentation. Accessed September 2026.

[^13]: PyTorch. [Reproducibility documentation, including scaled-dot-product attention backend differences](https://docs.pytorch.org/docs/stable/notes/randomness.html). PyTorch Documentation. Accessed September 2026.

[^14]: Woosuk Kwon et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). SOSP 2023.

### **Citation**

If you found this blog post helpful, please consider citing it:

```bibtex
@article{obasi2026understandingFlashAttention,
  title   = "Understanding FlashAttention: Personal Notes",
  author  = "Obasi, Chizoba",
  journal = "chizkidd.github.io",
  year    = "2026",
  month   = "Sep",
  url     = "https://chizkidd.github.io/2026/09/11/understanding-flashattention/"
}
```
