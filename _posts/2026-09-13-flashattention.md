---
layout: post
comments: true
title: "Understanding FlashAttention Pt 1: Personal Notes"
excerpt: "Personal annotations on an exact tiled-attention handbook: GPU memory traffic, online softmax, and the FlashAttention-1 through -4 family."
date: 2026-09-13
mathjax: true
---

## 0. Introduction

**How IO-Aware Attention Makes Transformers Faster Without Approximating Attention**

The mechanism, in three words: **Tiling + Online Softmax + Recomputation**. Everything in this handbook is elaboration on that summary.

> A technical handbook on exact tiled attention: GPU memory traffic, online softmax, forward and backward passes, IO complexity, the evolution from FlashAttention-1 through FlashAttention-4, and current framework behavior.

### 0.1 How to Read This Handbook

Before the fix, here is what the standard attention implementation looks like. Load $Q, K, V \in \mathbb{R}^{N \times d}$ in HBM, then:

1. Read $Q, K$ from HBM, compute $S$, write $S$ to HBM.
2. Read $S$ from HBM, compute $P$, write $P$ to HBM.
3. Read $P, V$ by blocks from HBM, compute $O$, write $O$ to HBM.
4. Return $O$.

What stands out to me is the number of round trips to HBM. Every intermediate value — $S$, $P$, $O$ — has to be written out and read back. That is the problem FlashAttention is solving.

The handbook itself frames the subject as easiest to understand when three different questions are kept separate:

> 1. What mathematical function is being computed? For dense attention, the target remains ordinary scaled dot-product attention.
> 2. How much arithmetic does that function require? Dense all-pairs query-key scoring remains quadratic in sequence length.
> 3. How does the implementation move data through the GPU memory hierarchy? This is where FlashAttention changes the algorithmic execution dramatically.

The central lesson I take from this framing is that wall-clock speed is not determined by FLOP count alone. An algorithm can perform essentially the same mathematical work, or even recompute intermediate values, and still run faster because it moves far less data to and from high-bandwidth memory.

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
P = \mathrm{softmax}\_{\mathrm{row}}(S), \quad O = PV.
$$

Throughout, HBM refers to large off-chip high-bandwidth GPU memory. On-chip memory is a broad teaching term for much smaller, faster storage such as registers and shared memory/SRAM. Exact hardware details vary by GPU generation.

The practical difference I keep coming back to:

| HBM | SRAM |
|---|---|
| slow | faster |
| large | smaller |
| off-chip | on-chip |

**System problem:** where do all those intermediate values $(S, P, O)$ live while the GPU computes them?

That question — not the arithmetic — is what FlashAttention was built to answer.

## Contents

### Part 1: The Fundamental Problem
- [1.1 What FlashAttention actually optimizes](#111-what-flashattention-actually-optimizes)
- [1.2 The attention equation is not the implementation](#112-the-attention-equation-is-not-the-implementation)
- [1.3 GPU memory hierarchy and why IO matters](#113-gpu-memory-hierarchy-and-why-io-matters)
- [1.4 Why materializing $S$ and $P$ is expensive](#114-why-materializing-s-and-p-is-expensive)
- [1.5 Dense arithmetic is still quadratic](#115-dense-arithmetic-is-still-quadratic)
- [1.6 Memory-efficient exact attention predates FlashAttention](#116-memory-efficient-exact-attention-predates-flashattention)

### Part 2: The Mathematical Trick
- [2.1 Tiling queries, keys, and values](#211-tiling-queries-keys-and-values)
- [2.2 Softmax is the difficult part of streaming](#212-softmax-is-the-difficult-part-of-streaming)
- [2.3 Online softmax from first principles](#213-online-softmax-from-first-principles)
- [2.4 The blockwise merge recurrence](#214-the-blockwise-merge-recurrence)
- [2.5 A complete numerical example](#215-a-complete-numerical-example)

### Part 3: Putting the Mathematics onto the GPU
- [3.1 FlashAttention forward pass](#311-flashattention-forward-pass)
- [3.2 Why dense FlashAttention is exact](#312-why-dense-flashattention-is-exact)
- [3.3 IO complexity: what the theorem actually says](#313-io-complexity-what-the-theorem-actually-says)
- [3.4 Memory complexity: linear auxiliary state, not linear compute](#314-memory-complexity-linear-auxiliary-state-not-linear-compute)
- [3.5 Causal masking and tile skipping](#315-causal-masking-and-tile-skipping)
- [3.6 Backward pass: recompute instead of save](#316-backward-pass-recompute-instead-of-save)
- [3.7 Why more FLOPs can still be faster](#317-why-more-flops-can-still-be-faster)

### Part 4: Architectural Compatibility
- [4.1 MHA, MQA, and GQA compatibility](#411-mha-mqa-and-gqa-compatibility)
- [4.2 Variable lengths, local attention, and dropout](#412-variable-lengths-local-attention-and-dropout)

### Part 5: FlashAttention Evolution
- [5.1 FlashAttention-2: what changed](#511-flashattention-2-what-changed)
- [5.2 FA2 parallelism across sequence tiles](#512-fa2-parallelism-across-sequence-tiles)
- [5.3 FA2 work partitioning and non-matmul FLOPs](#513-fa2-work-partitioning-and-non-matmul-flops)
- [5.4 FlashAttention-3: the Hopper generation](#514-flashattention-3-the-hopper-generation)
- [5.5 FA3 asynchrony: overlap data movement, GEMM, and softmax](#515-fa3-asynchrony-overlap-data-movement-gemm-and-softmax)
- [5.6 FA3 FP8: performance without pretending precision is free](#516-fa3-fp8-performance-without-pretending-precision-is-free)
- [5.7 FlashAttention-4: the Blackwell generation](#517-flashattention-4-the-blackwell-generation)
- [5.8 FA4 and asymmetric hardware scaling](#518-fa4-and-asymmetric-hardware-scaling)
- [5.9 FA4 implementation and current status](#519-fa4-implementation-and-current-status)
- [5.10 FlashAttention-1 through -4 compared](#5110-flashattention-1-through-4-compared)

### Part 6: Using FlashAttention in Frameworks
- [6.1 PyTorch scaled-dot-product attention today](#611-pytorch-scaled-dot-product-attention-today)
- [6.2 Exactness is not bitwise identity](#612-exactness-is-not-bitwise-identity)

### Part 7: FlashAttention vs. Other Techniques
- [7.1 FlashAttention vs. PagedAttention](#711-flashattention-vs-pagedattention)
- [7.2 FlashAttention vs. sparse and linear attention](#712-flashattention-vs-sparse-and-linear-attention)

### Part 8: Training vs. Inference
- [8.1 Training, prefill, and decode are different regimes](#811-training-prefill-and-decode-are-different-regimes)

### Part 9: Practical Engineering
- [9.1 Common implementation mistakes](#911-common-implementation-mistakes)
- [9.2 Common misconceptions and the practical mental model](#912-common-misconceptions-and-the-practical-mental-model)

---

## Part 1: The Fundamental Problem

### 1.1 What FlashAttention Actually Optimizes

Start with the ordinary attention function:

$$
O = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right) V.
$$

FlashAttention is **IO-aware**. My working definition:

Minimize data movement between the different levels of GPU memory, rather than just trying to reduce the number of mathematical operations (FLOPs).

The speed bottleneck in modern AI hardware is often not how fast the GPU can compute math, but how fast it can **read and write data**. This is the memory-compute tradeoff.

A textbook implementation often makes this look like three large operations:

> Form the score matrix $S$, apply row-wise softmax to obtain $P$, then multiply by $V$. Mathematically that is fine. On a GPU, however, writing a huge intermediate matrix to HBM and reading it back can be far more expensive than the equation suggests.

FlashAttention's central contribution is to make the algorithm IO-aware. It partitions the computation into tiles that fit in fast on-chip memory, streams blocks of $K$ and $V$, and maintains enough row-wise softmax state to produce the exact output without materializing the full $N \times N$ attention matrix in HBM.

> **What changes:** the execution schedule, memory traffic, and stored intermediates.
>
> **What does not change:** the dense scaled-dot-product attention function being evaluated.

This distinction is why "FlashAttention is a faster kind of attention" can be misleading. It is better thought of as an algorithm and kernel family for evaluating attention efficiently on accelerators. A model can use causal masking, RoPE, MQA/GQA, or other attention features and still use a FlashAttention implementation underneath.

The original paper contrasts this approach with approximate attention methods that reduce arithmetic by changing the mathematical problem. Dense FlashAttention does not make that trade. The same paper also introduced a block-sparse extension, but that sparse extension is a different case because omitting blocks changes which interactions are computed.

### 1.2 The Attention Equation Is Not the Implementation

The equation does not tell you where tensors live.

A **standard naive attention implementation**:

1. Calculate $S$, store $S$.
2. Read $S$, calculate $P$, store $P$.
3. Read $P$, calculate $O$.

The problem is the number of HBM round trips. A simple materializing implementation does:

$$
S \leftarrow QK^T / \sqrt{d}, \quad P \leftarrow \mathrm{softmax}(S), \quad O \leftarrow PV.
$$

If $S$ is written to HBM after the first matrix multiplication, read for softmax, $P$ is written back, and then $P$ is read again for the $PV$ multiplication, the GPU spends significant time moving an $N^2$ object through memory.

**FlashAttention avoids repeatedly moving huge intermediates:**

1. Calculate small $S$ tile, softmax tile, use tile with $V\_j$, discard tile.
2. Calculate next tile.

The key questions I ask when looking at any equation:

1. How many operations are required?
2. What data must move between memory levels to perform those operations?

FlashAttention reuses the same dependencies. Blocks of $Q$, $K$, and $V$ are brought near the compute units, score tiles are produced and consumed locally, and only compact row-wise statistics plus the output need to persist across tiles.

> **Algorithmic lesson.** A computational graph is not a memory schedule. Writing $P = \mathrm{softmax}(QK^T)$ on paper does not require an implementation to store all of $QK^T$ or $P$ in off-chip memory at once.

This idea generalizes beyond attention. Fused kernels, tiling, recomputation, and operator scheduling often trade a small amount of extra arithmetic for much less movement of large intermediates. On modern accelerators, that can be the right trade because matrix-multiply throughput has grown much faster than many other parts of the memory and execution hierarchy. FlashAttention-3 and -4 make that hardware dependence increasingly explicit.

### 1.3 GPU Memory Hierarchy and Why IO Matters

GPUs expose a hierarchy rather than one uniform pool of equally fast memory. The names and capacities vary by architecture, but the mental model is:

- Registers / very local state
- On-chip shared memory / SRAM
- HBM / device memory

HBM is large, but data must travel to the compute units. On-chip storage is much smaller, but reuse there is much cheaper. The original FlashAttention analysis models this asymmetry using HBM and SRAM and explicitly optimizes the number of transfers between them.

My breakdown of each level:

**HBM:**
- Relatively large
- Relatively slower to access
- Physically farther from individual compute operations
- Stores model weights, $Q/K/V$, activations, large tensors

**SRAM / shared memory:**
- Much smaller
- Faster
- Cheaper reuse cost

**Registers:**
- Even smaller
- More local

**Why tiling helps.** Suppose $Q\_i$ (a query tile) needs to interact with many $K/V$ tiles. Instead of constantly moving $Q\_i$ back and forth, we can keep it close to the compute units while processing:

$$
K\_1, V\_1 \to K\_2, V\_2 \to K\_3, V\_3 \to \cdots
$$

So one loaded $Q\_i$ can participate in lots of computation. This is called **reuse**.

> The purpose of tiling is not simply "make tensors smaller." It is to **increase reuse** while a tile is resident on chip. A block of $Q$ can interact with multiple $K/V$ blocks before its partial softmax/output state is written back. Conversely, $K/V$ blocks can be streamed through query blocks according to the chosen schedule.

> A kernel becomes IO-aware when the placement and movement of data are part of the algorithm, rather than an afterthought left to a sequence of separately launched tensor operations.

> Do not turn this into a universal slogan that attention is always "memory-bound." The bottleneck depends on sequence length, head dimension, dtype, mask pattern, GPU generation, forward vs. backward, and which kernel is running. FA3 and FA4 exist partly because, as hardware changed, the dominant bottlenecks changed too.

### 1.4 Why Materializing $S$ and $P$ Is Expensive

The quadratic intermediate becomes concrete very quickly. Suppose a batch contains one sequence, with 32 attention heads, sequence length $N = 8192$, and a two-byte dtype such as FP16 or BF16. One dense tensor with shape

$$
[1, 32, 8192, 8192]
$$

contains $32 \times 8192^2$ elements. At two bytes per element, that is

$$
32 \times 8192^2 \times 2 = 4{,}294{,}967{,}296 \text{ bytes} = 4 \text{ GiB}.
$$

That is the size of one full score- or probability-like tensor for this example.

Concretely, my arithmetic:

- Batch = 1
- Heads = 32
- seq_len = 8192
- dtype = FP16/BF16
- 2 bytes/elem
- tensor shape = $[1, 32, 8192, 8192]$
- $(32)(8192)(8192)$ elem $\times$ (2 bytes/elem) $= 4{,}294{,}967{,}296$ bytes $= 4$ GiB

Imagine reading and writing this much data.

A naive decomposition may produce arrays of this scale at multiple stages. This does not mean every modern framework keeps both $S$ and $P$ alive simultaneously, and compiler fusion can already avoid some traffic. The example illustrates the fundamental problem: a dense $N^2$ intermediate is large enough that repeatedly writing and rereading it can dominate memory use and bandwidth.

**Naive pipeline:**

$$
QK^T \to \text{huge } S \to \text{softmax} \to \text{huge } P \to PV
$$

**FlashAttention pipeline:**

$$
Q\_i K\_j^T \to \text{softmax tile} \to \text{tile} \times V\_j \to \text{discard}
$$

FlashAttention avoids storing the full matrix in HBM. It forms score tiles, applies the softmax update while the tile is on chip, immediately uses those probabilities to accumulate the corresponding contribution from $V$, then discards the tile.

> **Important wording:** FlashAttention removes the need to materialize the full attention matrix as an off-chip intermediate. It does not remove the logical pairwise interactions required by dense attention.

**Important distinction.** FlashAttention acknowledges that $N^2$ interactions exist and says: *do not store the entire result of these interactions as a giant intermediate if we can consume each piece immediately.*

This is also why the memory benefit is especially important during training, where naive autograd would otherwise want large intermediates for the backward pass.

### 1.5 Dense Arithmetic Is Still Quadratic

FlashAttention changes the memory schedule, not the mathematical function. The function is still dense attention. That means:

> For self-attention with $N$ tokens and head dimension $d$, forming all query-key scores requires work proportional to $N^2 d$.
>
> Multiplying the probabilities by values adds another dense pairwise matrix multiplication of the same broad order. FlashAttention reorganizes these operations, but it does not stop evaluating the dense set of query-key interactions.

So three statements must not be confused:

> - **Arithmetic complexity:** dense attention remains $\mathcal{O}(N^2 d)$.
> - **Large intermediate storage:** FlashAttention avoids an $\mathcal{O}(N^2)$ materialized score/probability tensor in HBM.
> - **HBM traffic:** the original paper proves a lower IO cost under its two-level memory model than standard materializing attention.

This is how a method can make much longer sequences practical without making long context "free." Doubling $N$ still roughly quadruples the number of dense query-key pairs. FlashAttention mainly attacks the data-movement and memory-footprint side of that computation.

> If you want to reduce the *number* of query-key pairs themselves, you need a different mathematical structure: for example sparsity, a local pattern, or a different attention formulation. Those choices can change model behavior and are conceptually separate from dense FlashAttention.

When reporting speedups, always distinguish asymptotic arithmetic from measured runtime. A kernel can become several times faster at the same $\mathcal{O}(N^2 d)$ complexity because the constant factors, occupancy, fusion, and memory traffic change dramatically.

My side-by-side summary:

| Quality | Dense FlashAttention |
|---|---|
| Arithmetic | $\mathcal{O}(N^2 d)$ |
| Full attention intermediates | Avoid $\mathcal{O}(N^2)$ storage |
| HBM traffic | Reduced substantially |

### 1.6 Memory-Efficient Exact Attention Predates FlashAttention

It would be historically inaccurate to say FlashAttention first discovered that exact attention can avoid quadratic memory.

> Rabe and Staats showed before FlashAttention that attention need not require $\mathcal{O}(N^2)$ memory with respect to sequence length. Their work gave exact memory-efficient algorithms while retaining quadratic time, and a practical accelerator implementation with subquadratic memory.

FlashAttention's distinct contribution was to turn memory efficiency into an explicit IO-aware GPU algorithm: tile the computation against the accelerator memory hierarchy, analyze HBM accesses, fuse the relevant operations, and show substantial wall-clock gains.

Online softmax also has an earlier lineage. Milakov and Gimelshein described an online recurrence that computes the classical stable softmax normalizer with fewer memory accesses. FlashAttention builds the same kind of running-max/running-normalizer idea into tiled attention, while also accumulating the value-weighted output.

> A precise lineage is therefore:
>
> - stable / online softmax provides a streaming normalization tool,
> - earlier memory-efficient attention shows exact attention need not store $N^2$ state,
> - FlashAttention co-designs tiling, softmax state, and GPU IO to make the approach fast in practice.

The lineage, in my shorthand:

$$
\text{Stable softmax} \to \text{Online normalization} \to \text{Exact memory-efficient attention} \to \text{FlashAttention} \to \text{FlashAttention-2} \to \text{FlashAttention-3} \to \text{FlashAttention-4}
$$

Rabe and Staats demonstrated exact memory-efficient attention with quadratic computation but subquadratic memory (the earlier work). There is also earlier work on online softmax by Milakov and Gimelshein.

FlashAttention's key contribution was to bring together:

1. Tiling
2. Online softmax
3. Fused computation
4. GPU memory hierarchy awareness
5. IO complexity analysis

into a practical high-performance algorithm.

This distinction matters because "memory-efficient" does not automatically mean "IO-optimized for a particular hardware model."

---

## Part 2: The Mathematical Trick

### 2.1 Tiling Queries, Keys, and Values

**Most important.** This is the section where the trick lives.

Suppose $Q \in \mathbb{R}^{N\_q \times d}$, $K \in \mathbb{R}^{N\_k \times d}$, and $V \in \mathbb{R}^{N\_k \times d\_v}$. Instead of processing everything at once, divide them into blocks. For example:

$$
Q: [Q\_1, Q\_2, Q\_3], \quad K, V: [K\_1, V\_1], [K\_2, V\_2], [K\_3, V\_3]
$$

For one pair,

$$
S\_{ij} \doteq \frac{Q\_i K\_j^T}{\sqrt{d}} + B\_{ij},
$$

we process that score tile locally.

Then, for each query block:

1. Calculate scores
2. Softmax them
3. Multiply by $V\_j$
4. Update the running output
5. Discard the tile

The softmax used at each tile is

$$
\mathrm{softmax}(x\_i) = \frac{e^{x\_i}}{\sum\_j e^{x\_j}}.
$$

**Matrix multiplication is easy to tile** because it is fundamentally accumulation:

$$
AB = \sum\_j A\_j B\_j,
$$

so we can calculate pieces and add them.

**Softmax is harder** because every element depends on the entire row. If we process the first block, we do not know the eventual denominator. Even worse, numerical stability requires knowing the maximum.

> Instead of forming all $N\_q N\_k$ scores at once, partition the matrices into blocks. For one query block $Q\_i$ and one key/value block $(K\_j, V\_j)$, compute
>
> $$
> S\_{ij} = \frac{Q\_i K\_j^T}{\sqrt{d}} + B\_{ij}.
> $$
>
> The score tile $S\_{ij}$ is small enough to be processed near the compute units. Its contribution is folded into running row statistics and an output accumulator, then the tile can be discarded.

Conceptually, for each query block:

> 1. load a tile of $Q\_i$,
> 2. stream compatible $K/V$ tiles,
> 3. compute one score tile,
> 4. update row-wise softmax state,
> 5. accumulate the corresponding $V$ contribution,
> 6. move to the next tile without writing a global $S$ or $P$ matrix.

Actual kernels choose tile shapes and loop order based on on-chip capacity, head dimension, GPU generation, causal structure, and work partitioning. The original FlashAttention IO analysis uses tile sizes derived from SRAM capacity $M$.

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

## Part 3: Putting the Mathematics onto the GPU

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

The exact theorem has assumptions, so do not interpret this as "FlashAttention always moves exactly this many bytes." It is an asymptotic result for a particular memory model.

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

The auxiliary attention state scales roughly linearly with sequence length, but linear memory is not linear computation. The computation is still

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

Consider autoregressive attention. Token $i$ cannot attend to future token $j > i$.

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

Conceptually: forward: don't store $P$. Backward: recompute $P$ tile-by-tile.

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

and then:

$$
dQ \mathrel{+}= dS \cdot K / \sqrt{d}
$$

$$
dK \mathrel{+}= dS^T \cdot Q / \sqrt{d}
$$

with appropriate masking. Where

$$
D\_i[:, \text{None}] = \sum\_j P\_{ij}\, dP\_{ij}
$$

is the column/vector broadcast across each row.

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

## Part 4: Architectural Compatibility

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

| Source § | Hierarchical |
|---|---|
| 1-6 | 1.1-1.6 |
| 7-11 | 2.1-2.5 |
| 12-18 | 3.1-3.7 |
| 19-20 | 4.1-4.2 |
| 21-30 | 5.1-5.10 |
| 31-32 | 6.1-6.2 |
| 33-34 | 7.1-7.2 |
| 35 | 8.1 |
| 36-37 | 9.1-9.2 |

