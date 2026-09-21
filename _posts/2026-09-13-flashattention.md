---
layout: post
comments: true
title: "FlashAttention: Part 1"
excerpt: "Notes on an exact tiled-attention handbook: GPU memory traffic, online softmax, FlashAttention-1, and architectural compatibility (MHA/MQA/GQA)."
date: 2026-09-13
mathjax: true
---

---

 _This is Part 1 of a two-part FlashAttention series. Check out [Part 2](https://chizkidd.github.io/2026/09/17/flashattention-2) next. Part 2 covers the FlashAttention evolution (FA-1/2/3/4)._

---

**How IO-Aware Attention Makes [Transformers](https://chizkidd.github.io/2026/04/17/transformers/) Faster Without Approximating Attention**

The [handbook](https://drive.google.com/file/d/1CLyK-9Cflcvi3fRl3qAHyzYvwJFjCyVg/view) of reference for this blogpost was inspired by this [tweet](https://x.com/techNmak/status/2098057360908685358) and is titled: _Understanding FlashAttention: How IO-Aware Attention Makes Transformers Faster Without Approximating Attention._ 

The mechanism, in three words: **Tiling + Online Softmax + Recomputation**. Everything in this handbook[^15] is elaboration on that summary. The handbook does a technical deep dive on exact tiled attention: _GPU memory traffic, online softmax, forward and backward passes, IO complexity, the evolution from FlashAttention-1 through FlashAttention-4, and current framework behavior._

In this blog post, we cover the fundamental problem FlashAttention-1 (FA1) addresses, the mathematical tricks utilised, the GPU implementation of these math tricks, and the architectural compatibility ([MHA](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#multi-head-attention)/[MQA](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#multi-query-attention-mqa)/[GQA](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#grouped-query-attention-gqa)) of FlashAttention.

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
- [1.4 Why materializing $S$ and $P$ is expensive](#14-why-materializing--and--is-expensive)
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
- [4.1 MHA, MQA, and GQA compatibility](#41-mha-mqa-and-gqa-compatibility-shared-kv-heads)
- [4.2 Variable lengths, local attention, and dropout](#42-variable-lengths-local-attention-and-dropout)

[5. Summary](#5-summary)

<!-- [5. FlashAttention Evolution](#5-flashattention-evolution)
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
- [9.2 Common misconceptions and the practical mental model](#92-common-misconceptions-and-the-practical-mental-model) -->


## **Appendix**
- [Reference Map](#reference-map)
- [References](#references)
- [Citation](#citation)

---

## **0. Introduction**

### 0.1 How to Read This Handbook

What stands out to me is the number of round trips to HBM in naive standard attention. Every intermediate value $(S$, $P$, $O)$ has to be written out and read back. That is the problem FlashAttention is solving. The handbook itself frames the subject as easiest to understand when three different questions are kept separate:

1. **What mathematical function is being computed?** For dense attention, the target remains ordinary scaled dot-product attention.

2. **How much arithmetic does that function require?** Dense all-pairs query-key scoring remains quadratic in sequence length.

3. **How does the implementation move data through the GPU memory hierarchy?** This is where FlashAttention changes the algorithmic execution dramatically.

The lesson I keep coming back to here: FLOP count alone doesn't decide wall-clock speed. An algorithm can do essentially the same math, or even recompute intermediates from scratch, and still finish faster just because it shuttles far less data to and from high-bandwidth memory (HBM).

<!-- > **Core distinction.** Dense FlashAttention is an *exact* attention algorithm: it does not replace softmax attention with a low-rank, sparse, kernelized, or approximate formula. "Exact" refers to the mathematical attention computation. Floating-point kernels can still differ by small rounding effects because operations are reordered.[^4] $^,$ [^10] $^,$ [^13] -->

{% capture c %}
Dense FlashAttention is an *exact* attention algorithm: it does not replace softmax attention with a low-rank, sparse, kernelized, or approximate formula. "Exact" refers to the mathematical attention computation. Floating-point kernels can still differ by small rounding effects because operations are reordered. $^{4, 10, 13}$
{% endcapture %}
{% include callout.html type="note" title="Core distinction" content=c %}

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

<!-- >**System problem:** Where do all those intermediate values $(S, P, O)$ live while the GPU computes them? -->

{% capture c %}
Where do all those intermediate values $(S, P, O)$ live while the GPU computes them?
{% endcapture %}
{% include callout.html type="note" title="System Problem" content=c %}

That question, not the arithmetic, is what FlashAttention was built to answer.

## **1. The Fundamental Problem**

### 1.1 What FlashAttention Actually Optimizes

Start with the ordinary attention function:

$$
O = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right) V.
$$

A textbook implementation boils this down to three big steps:

* Form the score matrix $S$, 
* apply row-wise softmax to get $P$, then 
* multiply by $V$

Mathematically, that's perfectly fine. **On a GPU, though, writing a giant intermediate to HBM and reading it back is often far more expensive than the equation lets on.**

FlashAttention's real contribution is an **IO-aware implementation**. It splits the computation into tiles that fit in fast on-chip memory, streams blocks of $K$ and $V$ past them, and keeps just enough row-wise softmax state around to produce the exact output. No full $N \times N$ attention matrix ever needs to land in HBM.[^4]

My working definition of **IO-awareness**: 

>minimize data movement between the different levels of GPU memory (HBM and SRAM), rather than just trying to reduce the number of mathematical operations (FLOPs). 

The reasoning behind it is that the speed bottleneck in modern AI hardware is often not how fast the GPU can compute math, but how fast it can **read and write data**. This is the memory-compute tradeoff.

**The problem, in one line:**

```text
compute → HBM write → HBM read → compute
```

The compute isn't the bottleneck. The round trips are. **Tiling** is the fix. Instead of one big $QK^T$, process it in blocks:

$$
Q K^T \rightarrow Q_i K_j^T \qquad (\text{for small blocks } Q_i, K_j)
$$

Each score block gets:

1. computed,
2. softmax-processed,
3. immediately multiplied by its corresponding $V_j$, and
4. discarded.

For $N = 8192$, instead of an $8192 \times 8192$ intermediate matrix, you process blocks like $128 \times 128$. All the $QK$ interactions still happen, just in manageable pieces.

<!-- > **What changes:** the execution schedule, memory traffic, and stored intermediates.
>
> **What does not change:** the dense scaled-dot-product attention function being evaluated. -->

{% capture c %}
**What changes:** the execution schedule, memory traffic, and stored intermediates.<br>
**What does not change:** the dense scaled-dot-product attention function being evaluated.
{% endcapture %}
{% include callout.html type="note" title="Important Fact" content=c %}

This framing matters. Calling FlashAttention "a faster kind of attention" misses the point. It's better understood as an algorithm and kernel family for **evaluating attention efficiently on accelerators.** A model can use causal masking, RoPE, [MQA](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#multi-query-attention-mqa)/[GQA](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#grouped-query-attention-gqa), or any other attention feature and still run a FlashAttention kernel underneath it.

The original paper draws a line between this and approximate attention methods, which cut arithmetic by changing the math itself. Dense FlashAttention doesn't do that. It keeps the math intact. The block-sparse extension from the same paper is a different story, since skipping blocks literally changes which query-key interactions get computed.[^4]

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

The **key questions I ask when looking at the attention equation:**
1. How many operations are required?<br>
2. What data must move between memory levels to perform those operations?

FlashAttention's move is simple: _keep the work close_. Blocks of $Q$, $K$, and $V$ sit near the compute units, score tiles get built and consumed on the spot, and only the running row statistics plus the output survive across tiles. Nothing else needs to stick around.

<!-- > **Algorithmic lesson.** A computational graph is not a memory schedule. Writing $P = \mathrm{softmax}(QK^T)$ on paper does not require an implementation to store all of $QK^T$ or $P$ in off-chip memory at once. -->

{% capture c %}
A computational graph is not a memory schedule. Writing $P = \mathrm{softmax}(QK^T)$ on paper does not require an implementation to store all of $QK^T$ or $P$ in off-chip memory at once.
{% endcapture %}
{% include callout.html type="note" title="Algorithmic lesson" content=c %}

The pattern isn't unique to attention. **Fused kernels, tiling, recomputation, operator scheduling**; they all do the same thing. Spend a little extra arithmetic to avoid dragging huge intermediates through memory. On modern hardware, that's usually a winning bet. Matmul throughput has raced ahead of the rest of the memory hierarchy, so the math is cheap and the data movement is what hurts. FlashAttention-3 (FA3) and FlashAttention-4 (FA4) take this further, and they're upfront about it: the algorithm bends to the hardware, not the other way around.[^6] $^,$ [^7]

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


FlashAttention-1 (FA1) models the HBM/SRAM asymmetry and explicitly optimizes the number of transfers between them.[^4] My breakdown of each GPU memory level in the figure above:

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
K_1, V_1 \to K_2, V_2 \to K_3, V_3 \to \cdots
$$

Therefore, one loaded $Q\_i$ can participate in lots of computation. This is called **reuse**. Tiling doesn't only make tensors smaller, it also **increases reuse** while a tile is resident on chip. Conversely, $K/V$ blocks can be streamed through query blocks according to the chosen schedule.

<!-- > A kernel becomes **IO-aware when the placement and movement of data are part of the algorithm,** rather than an afterthought left to a sequence of separately launched tensor operations. -->

{% capture c %}
A kernel becomes **IO-aware when the placement and movement of data are part of the algorithm,** rather than an afterthought left to a sequence of separately launched tensor operations.
{% endcapture %}
{% include callout.html type="note" title="IO-Awareness" content=c %}

This does not mean attention is "always memory bound." FA3 and FA4 exist partly because, as hardware changed, the dominant bottlenecks changed too.[^6] $^,$ [^7] The bottleneck depends on:
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

FlashAttention skips all of the naive attention decomposition. Score tiles get formed, softmaxed on chip, used to accumulate their contribution from $V$, then thrown away.[^4] The full matrix never touches HBM.

<div class="mermaid" style="margin: 2rem 0;">
flowchart LR
    QK["QᵢKⱼᵀ"] --> Softmax["softmax tile"]
    Softmax --> PV["tile × Vⱼ"]
    PV --> Discard["discard tile"]

    classDef boxed fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,font-size:22px
    classDef plain fill:none,stroke:none,font-size:26px,font-weight:bold

    class QK,Softmax,PV boxed
    class Discard plain
</div>

<small class="text-muted d-block text-center">**Figure 3:** FlashAttention Implementation Pipeline.</small>

<!-- > **Important wording:** FlashAttention removes the need to materialize the full attention matrix as an off-chip intermediate. It does not remove the logical pairwise interactions required by dense attention. -->

{% capture c %}
FlashAttention removes the need to materialize the full attention matrix as an off-chip intermediate. It does not remove the logical pairwise interactions required by dense attention.
{% endcapture %}
{% include callout.html type="note" title="Important wording" content=c %}

This matters most during training, where naive autograd would want those large intermediates saved for the backward pass.

**Important distinction:** FlashAttention acknowledges that $N^2$ interactions exist and says: *Do not store the entire result of these interactions as a giant intermediate if we can consume each piece immediately.*

### 1.5 Dense Arithmetic Is Still Quadratic

FlashAttention changes the memory schedule, not the math. The function itself is still dense attention. That means forming all query-key scores for $N$ tokens with head dimension $d$ still costs work proportional to $N^2 d$. There's no way around that.

Here's the thing about multiplying probabilities by values: it's another dense pairwise matrix multiplication of the same broad order. FlashAttention reshuffles these operations, but it doesn't skip the dense set of query-key interactions. Those still happen.[^4] $^,$ [^5]

So three things must not get conflated:

- **Arithmetic complexity:** dense attention stays at $O(N^2 d)$.
- **Large intermediate storage:** FlashAttention avoids an $O(N^2)$ materialized score/probability tensor in HBM.
- **HBM traffic:** the original paper proves a lower IO cost than standard materializing attention under its two-level memory model.[^4]

This is how a method makes much longer sequences practical without making long context "free." Double $N$, and you roughly quadruple the number of dense query-key pairs. FlashAttention just attacks the data-movement and memory-footprint side of that.

<!-- > If you actually want to reduce the *number* of query-key pairs, you need a different mathematical structure: sparsity, a local pattern, or a different attention formulation. Those choices change model behavior, and they're conceptually separate from dense FlashAttention. -->

{% capture c %}
If you actually want to reduce the *number* of query-key pairs, you need a different mathematical structure: sparsity, a local pattern, or a different attention formulation. Those choices change model behavior, and they're conceptually separate from dense FlashAttention.
{% endcapture %}
{% include callout.html type="note" title="Important Fact" content=c %}

When you report speedups, always separate asymptotic arithmetic from measured runtime. A kernel can run several times faster at the same $O(N^2 d)$ complexity because the **constant factors, occupancy, fusion, and memory traffic change dramatically.**

Essentially, the key takeaways are summarized below:

| Quality | Dense FlashAttention |
|---|---|
| Arithmetic | $O(N^2 d)$ |
| Full attention intermediates | Avoid $O(N^2)$ storage |
| HBM traffic | Reduced substantially |

### 1.6 Memory-Efficient Exact Attention Predates FlashAttention

It would be historically inaccurate to say FlashAttention first discovered that exact attention can avoid quadratic memory. Earlier work by **Rabe and Staats** demonstrated exact memory-efficient attention with quadratic computation but subquadratic memory. They showed that attention does not require $O(N^2)$ memory with respect to $N$.[^3] **Milakov and Gimelshein** also performed earlier work on online softmax that showed a memory-efficient, online recurrent methodology of computing the classical stable softmax normalizer.[^2] FlashAttention builds the same kind of running-max/running-normalizer idea into tiled attention, while also accumulating the value-weighted output.[^4]
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

## **2. The Mathematical Trick**

### 2.1 Tiling Queries, Keys, and Values

**Most important.** This is the section where the trick lives. Score tiles are small enough to stay near the compute units. Each tile's contribution gets folded into the running row statistics and the output accumulator, then the tile is thrown away.

- Suppose $Q \in \mathbb{R}^{N\_q \times d}$, $K \in \mathbb{R}^{N\_k \times d}$, and $V \in \mathbb{R}^{N\_k \times d\_v}$. Instead of processing everything $N_q, N_k$ at once, divide them into blocks. For example:

    $$
    Q: \boxed{Q_1} \boxed{Q_2} \boxed{Q_3} \quad K, V: \boxed{K_1, V_1} \boxed{K_2, V_2} \boxed{K_3, V_3}
    $$

    For one query-key/value pair ($Q_i, K_j/V_j$), we process that score tile $S_{ij}$ locally:

    $$
    S_{ij} \doteq \frac{Q_i K_j^T}{\sqrt{d}} + B_{ij},
    $$

    Then, for each query block:

    ```text
    1. Calculate score tile, S_ij
    2. Row-wise softmax and update
    3. Multiply by V_j and accumulate
    4. Update the running output
    5. Discard the tile 
    6. Move to the next tile while ignoring the need to write a global S or P matrix.
    ```

    The softmax used at each tile is

    $$
    \mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}.
    $$

- **Matrix multiplication is easy to tile** because it is fundamentally accumulation:

    $$
    AB = \sum_j A_j B_j,
    $$

    so we can calculate pieces and add them.

- **Softmax is harder** because every element depends on the entire row. If we process the first block, we do not know the eventual denominator. Even worse, numerical stability requires knowing the maximum.

- For kernel executions, we choose tile shapes and loop order based on: **on-chip capacity, head dimension, GPU generation, causal structure, and work partitioning.** FA1 uses tile sizes derived from SRAM capacity $M$.[^4]

<!-- > Tiling alone is not enough. Matrix multiplication tiles compose naturally because sums can be accumulated. Softmax couples every score in a row through a shared maximum and denominator, so we need a way to merge blocks without seeing the full row at once. -->

{% capture c %}
Tiling alone is not enough. Matrix multiplication tiles compose naturally because sums can be accumulated. Softmax couples every score in a row through a shared maximum and denominator, so we need a way to merge blocks without seeing the full row at once.
{% endcapture %}
{% include callout.html type="note" title="Mathematical Trick" content=c %}

### 2.2 Softmax Is the Difficult Part of Streaming

A numerically stable softmax for one row $x\_1, \ldots, x\_N$ uses

$$
m = \max_j x_j, \qquad \ell = \sum_j e^{x_j - m}, \qquad p_j = \frac{e^{x_j - m}}{\ell}.
$$

Subtracting $m$ keeps large logits from blowing up. However, it seems to create a streaming problem: **how does one normalize an early block if a later block turns out to have a bigger maximum?**

Keep the running stats around, and **rescale** them when the maximum changes. Say the state after the earlier elements is $(m\_{\text{old}}, \ell\_{\text{old}})$, and the new block's maximum is $m\_b$. Then

$$
m_{\text{new}} = \max(m_{\text{old}}, m_b)
$$

Anything accumulated under the old max converts to the new one by a single factor:

$$
\alpha \doteq e^{m_{\text{old}} - m_{\text{new}}}
$$

The new block is then evaluated against the same $m\_{\text{new}}$.

None of this is an approximation. It's just the identity

$$
e^{x - m_{\text{old}}}\, e^{m_{\text{old}} - m_{\text{new}}} = e^{x - m_{\text{new}}}
$$

<!-- >**Mathematical insight.** The running maximum is a change of numerical reference point. When that reference changes, previously accumulated exponentials can be rescaled exactly in real arithmetic rather than recomputed from scratch. -->

{% capture c %}
The running maximum is a change of numerical reference point. When that reference changes, previously accumulated exponentials can be rescaled exactly in real arithmetic rather than recomputed from scratch.
{% endcapture %}
{% include callout.html type="note" title="Mathematical Insight" content=c %}

The online-normalizer recurrence predates FlashAttention and is the mathematical basis for streaming stable softmax.[^2]


### Worked example 1. 

Start with two extreme values to see why the max subtraction matters:

$$
\begin{aligned}
x &= [1000, 999], \quad \text{Directly computing $e^{1000}$ overflows}\\
m &= 1000 \\
x - m &= [1000 - 1000, \; 999 - 1000] = [0, -1] \\
e^[0, -1] &\approx [1, 0.368] \\
\end{aligned}
$$

The exponential values $1$ and $0.368$ are perfectly manageable.

<!-- $$
x = [1000, 999].
$$

Directly computing $e^{1000}$ overflows. Instead, set $m = 1000$, so

$$
x - m = [1000 - 1000, \; 999 - 1000] = [0, -1],
$$

and $e^{0} = 1$, $e^{-1} \approx 0.368$ are perfectly manageable. -->

### Worked example 2.

Now let's work through streaming two blocks.

$$
\begin{aligned}
\text{Block 1}: [2, 1]&, \text{ Block 2}:[4, 3]\\
\text{After Block 1} &: m_{\text{old}} = 2 \\
\text{Evaluating Block 1 at } m_{\text{old}} &: [e^{0}, e^{-1}] \\
\text{When Block 2 arrives} &: m_b = 4 \\
\text{Now in Block 2} &: m_{\text{new}} = \max(2, 4) = 4, \quad \alpha = e^{2 - 4} = e^{-2}\\
\text{Rescaling Block 1 at the new reference} &: [\alpha e^{0}, \; \alpha e^{-1}] = [e^{-2}, \; e^{-3}] \\
\text{Block 2 at the new reference} &: [e^{4 - 4}, \; e^{3 - 4}] = [e^{0}, \; e^{-1}] \\
\end{aligned}
$$

Putting it all together, with $x = [2, 1, 4, 3]$:

$$
\begin{aligned}
\mathrm{softmax}(x) &= \frac{[e^{-2}, \; e^{-3}, \; e^{0}, \; e^{-1}]}{e^{-2} + e^{-3} + e^{0} + e^{-1}}\\
\text{Block 1 streaming computation} &: \ell = \sum_j e^{x_j - m} = e^{0} + e^{-1} = 1 + e^{-1}, \\
\text{Block 1 normalized output} &: \frac{[e^{0}, e^{-1}]}{(1 + e^{-1})} \\
\text{Block 2 running normalizer} &: \ell_{\text{new}} = \alpha\, \ell_{\text{old}} + \sum_{j \in \text{Block 2}} e^{x_j - m_{\text{new}}} = e^{-2}(1 + e^{-1}) + e^{0} + e^{-1} \\
\text{Block 2 unnormalized accumulator} &: a = [\alpha e^{0}, \alpha e^{-1}, e^{0}, e^{-1}] = [e^{-2}, e^{-3}, e^{0}, e^{-1}] \\
\text{Block 1 normalized output} &: \frac{[e^{-2}, e^{-3}, e^{0}, e^{-1}]}{(e^{-2} + e^{-3} + 1 + e^{-1})} \
\end{aligned}
$$

This matches the full softmax computed in one shot.

### 2.3 Online Softmax from First Principles

Process scalar logits $x\_1, x\_2, \ldots$ one at a time. Initialize

$$
m_0 = -\infty, \qquad \ell_0 = 0
$$

After observing $x\_j$:

$$
m_j = \max(m_{j-1}, x_j), \qquad \ell_j = \ell_{j-1} e^{m_{j-1} - m_j} + e^{x_j - m_j}
$$

Milakov and Gimelshein showed this produces the same stable softmax normalizer as the conventional safe-softmax procedure, just with fewer passes over the input.[^2]

Attention needs a weighted value sum. So we need to maintain a value-weighted, unnormalized accumulator:

$$
a = \sum_j e^{x_j - m} v_j
$$

When the maximum moves from $m$ to $m'$, the accumulated statistics shift to the new reference before anything new gets added:

$$
a_{\text{old}} \to a_{\text{old}}\, e^{m_{\text{old}} - m_{\text{new}}}, \qquad \ell_j \to \ell_{j-1}\, e^{m_{j-1} - m_j} + e^{x_j - m_j}
$$

Then the new value contributions get folded in. At the end:

$$
o = \frac{a}{\ell}
$$

<!-- > For attention, $x\_j$ is not a fixed input vector stored in advance. Each block of logits is generated on demand from a matrix product $QK\_j^T / \sqrt{d}$ plus mask/bias terms. The online recurrence lets the kernel consume that block immediately. -->

{% capture c %}
For attention, $x\_j$ is not a fixed input vector stored in advance. Each block of logits is generated on demand from a matrix product $QK\_j^T / \sqrt{d}$ plus mask/bias terms. The online recurrence lets the kernel consume that block immediately.
{% endcapture %}
{% include callout.html type="note" title="Important Fact" content=c %}

The same idea works row by row and block by block, which is what makes a tiled exact softmax-attention forward pass possible.

**Key Insight.** We don't need the entire probability vector. We only need enough information to reconstruct its contribution to the final output.

<!-- {% capture c %}
We don't need the entire probability vector. We only need enough information to reconstruct its contribution to the final output.
{% endcapture %}
{% include callout.html type="note" title="Mathematical insight II" content=c %} -->

### 2.4 The Blockwise Merge Recurrence

For one query row, suppose the running state after some key blocks is

$$
(m, \ell, a),
$$

where $m$ is the running maximum score, $\ell$ is the stable softmax normalizer under that maximum, and $a \in \mathbb{R}^{d\_v}$ is the unnormalized value accumulator.

A new block lands: scores $s \in \mathbb{R}^b$, values $V\_b \in \mathbb{R}^{b \times d\_v}$. First, find the block's own maximum and update the running one:

$$
m_b = \max(s), \qquad m' = \max(m, m_b).
$$

The rescale factor and the block's exponentials follow:

$$
\alpha = e^{m - m'}, \qquad p = e^{s - m'}.
$$

Then update:

$$
\ell' = \alpha \ell + \sum_j p_j, \qquad a' = \alpha a + p^T V_b,
$$

$$
m \leftarrow m', \qquad \ell \leftarrow \ell', \qquad a \leftarrow a'.
$$

At the end:

$$
o = a / \ell.
$$

When you process a block of query rows, not just one, the running state changes shape. $m$ and $\ell$ each become vectors, one entry per row, and $a$ becomes a matrix. If you have masks, apply them to the score tile before the exponentials. Masked positions contribute zero probability, so they drop out of the update entirely. FA1 and FA2 express equivalent running-max/running normalizer/output updates in block form.[^4] $^,$ [^5]

<!-- > The recurrence is the algebraic reason tile boundaries do not change the dense softmax result. A different tiling changes the order of floating-point operations, but not the intended real-arithmetic function. -->

{% capture c %}
The recurrence is the algebraic reason tile boundaries do not change the dense softmax result. A different tiling changes the order of floating-point operations, but not the intended real-arithmetic function.
{% endcapture %}
{% include callout.html type="note" title="Important Fact" content=c %}

**This recurrence is what makes tiled exact attention possible.**

<!-- The mean of the running state:

- $m$ = running maximum score 
- $\ell$ = stable softmax normalizer
- $a$ = unnormalized value accumulator

The blockwise recurrence in my own notation:

- For one query row, maintain $(m, \ell, a)$.
- Suppose the next score block is $s = [s\_1, s\_2, \ldots, s\_b]$ with corresponding values $V\_b \in \mathbb{R}^{b \times d\_v}$.
- First calculate the block maximum: $m\_b = \max(s)$, 
- Then update the global maximum: $m' = \max(m, m\_b)$.
- Define: $\alpha = e^{m - m'}$, $p = e^{s - m'}$.
- Then update: 
    - $\ell' = \alpha \ell + \sum\_j p\_j$, $a' = \alpha a + p^T V\_b$.
    - $m \leftarrow m'$, $\ell \leftarrow \ell'$, $a \leftarrow a'$.
- Finally at the end: $o = a / \ell$. -->

### 2.5 A Complete Numerical Example

Take one already-scaled, unmasked attention row

$$
s = [2, 1, 4, 3]
$$

with two-dimensional values

$$
v\_1 = [1, 0], \quad v\_2 = [0, 1], \quad v\_3 = [2, 0], \quad v\_4 = [0, 2].
$$

The global maximum is 4, so the stable unnormalized weights are

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

Now run it through in two blocks, the way a tiled kernel would: $[2, 1]$ and $[4, 3]$.

**Block 1: $s = [2, 1]$.** Running max $m\_1 = 2$, stable exponentials $[e^{0}, e^{-1}] = [1, e^{-1}]$. So

$$
\ell\_1 = 1 + e^{-1} \approx 1.36788,
$$

and the value accumulator comes out to

$$
a\_1 = 1 \cdot [1, 0] + e^{-1} \cdot [0, 1] = [1, e^{-1}].
$$

**Block 2: $s = [4, 3]$.** The block max is $m\_b = 4$, beating the running max, so $m' = \max(2, 4) = 4$. The rescale factor:

$$
\alpha = e^{2 - 4} = e^{-2}.
$$

Then

$$
\ell\_2 = \alpha \ell\_1 + e^{0} + e^{-1} = e^{-2}(1 + e^{-1}) + 1 + e^{-1} = e^{-2} + e^{-3} + 1 + e^{-1} \approx 1.553,
$$

and

$$
a\_2 = e^{-2} \cdot [1, e^{-1}] + [2 e^{0}, 2 e^{-1}] = [2 + e^{-2}, 2 e^{-1} + e^{-3}] \approx [2.135, 0.7855].
$$

**Output:**

$$
O\_2 = a\_2 / \ell\_2 \approx [1.375, 0.506],
$$

matching the full-row computation.

<!-- > The old block was not revisited. Its contribution was merely rescaled when a larger maximum appeared. -->

{% capture c %}
The old block was not revisited. Its contribution was merely rescaled when a larger maximum appeared.
{% endcapture %}
{% include callout.html type="note" title="Important Fact" content=c %}

**This rescale-instead-of-recompute is what makes streaming possible.**

---

## **3. Putting the Mathematics onto the GPU**

### 3.1 FlashAttention Forward Pass

The forward pass, conceptually:

1. Partition $Q$ into query-row tiles, and $K$ and $V$ into key/value tiles.
2. For each query tile, initialize row-wise running maxima, normalizers, and output accumulators: $m = -\infty$, $\ell = 0$, $a = 0$.
3. Load a $K\_j, V\_j$ tile and form the local score tile $S\_{ij} = Q\_i K\_j^T / \sqrt{d}$.
4. Apply masks or biases $B\_{ij} / M\_{ij}$ belonging to this tile.
5. Find the tile maximum, update the running maximum $\max(m, m\_b)$, and rescale the previous state.
6. Exponentiate the current tile relative to the updated maximum.
7. Update the normalizer $\ell$ and the value-weighted accumulator $a$.
8. Repeat for every required key tile.
9. Normalize the accumulator row-wise $O = a / \ell$, then write the output.

The original algorithm picks block sizes so the tiles and running state fit in on-chip SRAM, cutting trips to HBM.[^4] A quick summary of the forward pass is shown below:

```python
for each K/V block:
    scores = Q @ K_block.T / sqrt(d)
    update running softmax, l
    update output accumulator, a
    calculate the normalized output, O
```

You can reproduce the algebra in a few lines of PyTorch, but that's not the same as a high-performance FlashAttention kernel. Real production implementations additionally exploit **GPU-specific tiling, registers, shared memory, warp scheduling, tensor cores, async copies, specialized intrinsics, and occupancy optimization.**

<!-- > The essential algorithmic idea is independent of one CUDA kernel: generate a score tile, consume it immediately through online softmax and $V$ accumulation, and never materialize the complete score/probability matrix in HBM. -->

{% capture c %}
The essential algorithmic idea is independent of one CUDA kernel: generate a score tile, consume it immediately through online softmax and $V$ accumulation, and never materialize the complete score/probability matrix in HBM.
{% endcapture %}
{% include callout.html type="note" title="Important Fact" content=c %}

**Minimal educational PyTorch.**

```python
# Single query block, tiled over K/V. Production kernels also
# tile Q; the outer loop is omitted here to focus on the recurrence.
def tiled_attention(q, k, v, block=128):
    scale = 1 / math.sqrt(q.shape[-1])
    m = q.new_full((q.shape[0],), -float("inf"))
    l = q.new_zeros(q.shape[0])
    a = q.new_zeros((q.shape[0], v.shape[-1]))

    for j in range(0, k.shape[0], block):
        kj, vj = k[j:j+block], v[j:j+block]
        s = (q @ kj.T) * scale
        m_new = torch.maximum(m, s.max(dim=-1).values)
        alpha = torch.exp(m - m_new)
        p = torch.exp(s - m_new[:, None])
        a = a * alpha[:, None] + p @ vj
        l = l * alpha + p.sum(dim=-1)
        m = m_new

    return a / l[:, None]
```

<!-- ```python
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
``` -->

That production-educational code gap matters a lot for FA-2, FA-3, and FA-4. Every generation keeps the same semantic structure, but schedules the work differently for newer hardware.[^5] $^,$ [^6] $^,$ [^7] Same math, different schedule, each tuned to the hardware of its generation.


### 3.2 Why Dense FlashAttention Is Exact

Does FlashAttention approximate attention? For dense FlashAttention, **NO**. The target function is untouched:

$$
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right) V.
$$

A short checklist for why this counts as exact:

- No Q-K pairs are intentionally removed.
- No low-rank approximation is introduced.
- No alternative kernelized attention function replaces softmax.
- Tiling just processes the same interactions in blocks.

Online softmax doesn't swap the exponential or normalization for something else either. The block recurrence only changes the order in which sufficient statistics get accumulated.

But *exact* needs a qualification. Suppose two implementations compute $a + b + c$:

- **A:** $(a + b) + c$
- **B:** $a + (b + c)$

In exact mathematics,

$$
(a + b) + c = a + (b + c).
$$

In floating point, in some cases,

$$
(a + b) + c \neq a + (b + c).
$$

Tiling changes the reduction order. Fusion can change how rounding shakes out. So:

$$
\text{exact algorithm} \neq \text{bitwise identical output}.
$$
Three caveats keep the word *exact* precise:

- **Floating point has finite precision.** Reorder the additions and reductions and you'll get small numerical differences from another implementation. PyTorch says so directly: SDPA backends can differ because floating-point ops get fused and ordered differently.[^10] $^,$ [^13]

- **Dropout is random during training.** If you want two runs to match bit-for-bit, you have to line up the random behavior too. Attention semantics alone won't be enough.

- **Sparse variants aren't dense.** The block-sparse variant in the original paper skips blocks entirely, so it approximates full dense attention rather than reproducing it.[^4]

<!-- > "Exact" doesn't mean "bitwise identical to every reference kernel." It means the algorithm isn't quietly changing dense softmax attention to save work. -->

{% capture c %}
_"Exact"_ doesn't mean _"bitwise identical to every reference kernel."_ It means the algorithm isn't quietly changing dense softmax attention to save work.
{% endcapture %}
{% include callout.html type="note" title="Important Fact" content=c %}

This distinction becomes important in numerical testing. A sensible tolerance depends on **dtype, accumulation order, sequence length, and backend, rather than binary identity as a pre-requisite.** 


<!-- Does FlashAttention approximate attention? For dense FlashAttention, **no**. The target remains

$$
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right) V.
$$

My checklist for why this is exact:

- No Q-K pairs are intentionally removed.
- No low-rank approximation is introduced.
- No alternative kernelized attention function replaces softmax.
- This algorithm simply processes the same interactions in blocks.

But *exact* has a qualification. Let me illustrate. Suppose two implementations compute $a + b + c$:

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

This distinction matters when evaluating numerical tests. A sensible tolerance depends on dtype, accumulation order, sequence length, and backend rather than requiring binary identity. -->

### 3.3 IO Complexity: What the Theorem Actually Says

The original FlashAttention paper works in a two-level memory model: HBM, plus on-chip SRAM of size $M$. The key assumptions are head dimension $d$ and $d \leq M \leq Nd$. In that regime, the two attention types require:

- **Standard materializing attention:** $\Theta(Nd + N^2)$ HBM accesses
- **FlashAttention:** $\Theta\left(\frac{N^2 d^2}{M}\right)$ HBM accesses

That second quantity is a **communication-complexity bound.** It counts scalar movement between modeled memory levels, up to asymptotic factors. It is not a byte count, and it is not the arithmetic complexity, dense attention still performs $O(N^2 d)$ work.

A few things worth keeping straight:

- These IO-complexity results hold for a specific memory model, not every GPU.
- The bound tracks movement of scalar elements between modeled memory levels, up to asymptotic factors.
- It is not the arithmetic complexity. Dense attention still performs $O(N^2 d)$ work.
- More usable on-chip memory $M$ means more reuse and less modeled HBM traffic.

The intuition: bigger on-chip memory lets larger working tiles stay resident. Bigger tiles mean more reuse and fewer HBM reloads. In the theorem's regime, that reuse is exactly why the HBM-access bound shrinks as $M$ grows. It's also why GPU architecture matters so much to FlashAttention performance.

<!-- > A common incorrect summary is "FlashAttention reduces attention IO from $O(N^2)$ to $O(N)$." The linear quantity is the large **auxiliary memory footprint with respect to sequence length,** not the general HBM-access expression above. -->

{% capture c %}
A common incorrect summary is "FlashAttention reduces attention IO from $O(N^2)$ to $O(N)$." The linear quantity is the large **auxiliary memory footprint with respect to sequence length,** not the general HBM-access expression above.
{% endcapture %}
{% include callout.html type="note" title="Important Fact" content=c %}


### 3.4 Memory Complexity: Linear Auxiliary State, Not Linear Compute

FlashAttention papers often say memory becomes linear instead of quadratic. That's true, but it's easy to misread, so it's worth being precise about what's linear and what isn't:

$$
\text{linear auxiliary state} \neq \text{linear compute.}
$$

A naive attention implementation creates $O(N^2)$ intermediates. FlashAttention doesn't. It keeps only:

- $Q/K/V$ tiles
- row-wise $m$
- row-wise $\ell$
- output accumulator $a$

The inputs and output already contain $O(Nd)$ elements on their own. A materializing dense attention implementation additionally creates $O(N^2)$ score and probability state. FlashAttention skips those full matrices and holds only the tiled working state plus row-wise statistics. So the extra memory tied to the attention operation scales as $O(Nd)$, linear in sequence length for a fixed head dimension $d$. But linear memory is not linear compute. The computation is still $O(N^2 d)$.

**Example.** For $N = 10{,}000$, there are roughly $10{,}000^2 = 100{,}000{,}000$ Q-K interactions. FlashAttention doesn't make those disappear. It just prevents you from needing a giant intermediate holding all of them at once.

Training is where this difference bites hardest. A straightforward backward pass would otherwise want the full probability matrix $P$ kept around. FlashAttention stores compact info instead, output plus row-wise normalization stats, and recomputes score and probability tiles during the backward pass.

<!-- > **Linear memory does not imply linear runtime.** The number of dense query-key interactions is still quadratic in $N$. -->

{% capture c %}
**Linear memory does not imply linear runtime.** The number of dense query-key interactions is still quadratic in $N$.
{% endcapture %}
{% include callout.html type="note" title="Important Fact" content=c %}

Two more things to keep in mind. First, don't claim that total model memory is $O(N)$. Other Transformer components consume activation memory, and autoregressive serving has KV-cache memory that grows with context length. FlashAttention is changing how the attention computation manages its intermediates, not the whole model's memory profile. Second, FlashAttention and activation checkpointing can coexist, since both trade recomputation for reduced stored state, just at different scopes of the training graph.


### 3.5 Causal Masking and Tile Skipping

In autoregressive attention, query token $i$ can't attend to future key token $j > i$. The attention matrix looks like

$$
\begin{bmatrix}
\checkmark & \times & \times & \times \\
\checkmark & \checkmark & \times & \times \\
\checkmark & \checkmark & \checkmark & \times \\
\checkmark & \checkmark & \checkmark & \checkmark
\end{bmatrix}
$$

with mask

$$
M = \begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0 
\end{bmatrix}
$$

Apply the mask before softmax and you get

$$
\mathrm{softmax}(S') = \begin{bmatrix}
\text{value} & 0 & 0 & 0 \\
\text{value} & \text{value} & 0 & 0 \\
\text{value} & \text{value} & \text{value} & 0 \\
\text{value} & \text{value} & \text{value} & \text{value} 
\end{bmatrix}
$$

A materialized mask is another $N \times N$ object. But a tiled kernel can reason about the geometry of each tile directly:

- Blocks strictly above the causal boundary are fully masked, so they can be skipped.
- Blocks strictly below the boundary are fully valid.
- Only blocks intersecting the diagonal need element-level causal masking.

That's a lot of invalid tiles you never have to compute. FlashAttention-2 explicitly exploits this structure, and current implementations also define alignment rules for unequal query/key lengths.

<!-- > Mask semantics are an API detail that must not be guessed. For example, current PyTorch SDPA treats a Boolean `attn_mask` value of `True` as a position that *participates* in attention, while other PyTorch mask APIs use different conventions. -->

{% capture c %}
Mask semantics are an API detail that must not be guessed. For example, current PyTorch SDPA treats a Boolean `attn_mask` value of `True` as a position that *participates* in attention, while other PyTorch mask APIs use different conventions.
{% endcapture %}
{% include callout.html type="note" title="Important Fact" content=c %}

**Connection to local attention.** Sliding-window attention works on the same principle. If each token only attends to 128 nearby tokens, most tiles can be skipped the same way. But this is a different move: you've changed the mathematical attention pattern itself. The model is now doing sparse or local attention, and FlashAttention is just the kernel executing that pattern. It's no longer the same problem as unrestricted dense attention.


### 3.6 Backward Pass: Recompute Instead of Save

A naive training implementation saves $P = \mathrm{softmax}(S)$ for the backward pass. That's an $N^2$ tensor sitting in HBM. FlashAttention doesn't keep it. It stores compact row-wise normalization information instead, then recomputes score and probability tiles when gradients are needed.

**Forward:**

- Store the output and row-wise normalization statistics.
- Do not store $P \in \mathbb{R}^{N \times N}$.

**Backward:**

- For each tile: recompute $QK^T$, reconstruct the local probabilities, compute gradients, discard the tile.
- Same trade as before: more arithmetic, less memory traffic.

For

$$
S = QK^T / \sqrt{d}, \quad P = \mathrm{softmax}(S), \quad O = PV,
$$

the backward pass relies on a row-wise identity:

$$
D_i = \sum_r dO_{ir}\, O_{ir} = \sum_j P_{ij}\, dP_{ij}
$$

After recomputing a tile of $P$, the local derivatives fall out as

$$
dV \mathrel{+}= P^T dO, \qquad dP = dO\, V^T, \qquad dS = P \odot (dP - D_i[\text{:}, \text{None}]),
$$

and then

$$
dQ \mathrel{+}= dS\, K / \sqrt{d}, \qquad dK \mathrel{+}= dS^T\, Q / \sqrt{d}.
$$

Masks contribute zero probability and zero gradient. FA2 stores a row-wise log-sum-exp quantity so the probability tile can be reconstructed stably from recomputed scores.

<!-- > Backward recomputation is intentional. It spends extra matrix-multiply work to avoid reading and writing a giant probability tensor, which is a favorable trade on GPUs. -->

{% capture c %}
Backward recomputation is intentional. It spends extra matrix-multiply work to avoid reading and writing a giant probability tensor, which is a favorable trade on GPUs.
{% endcapture %}
{% include callout.html type="note" title="Important Fact" content=c %}

This is the **memory-compute tradeoff** applied to the backward pass.

### 3.7 Why More FLOPs Can Still Be Faster

This is one of the biggest lessons from FlashAttention. The usual assumption is that fewer FLOPs means faster. On a GPU, that's not always true.

Different operations run at completely different throughputs. Tensor cores chew through matrix multiplication. Everything else is comparatively expensive: exponentials, reductions, synchronization, shared-memory operations, HBM transfers.[^4] $^,$ [^5] ***Therefore, doing extra arithmetic can be the right move if it kills expensive memory traffic.***

Consider two options:

- **A:** compute, write huge $P$ to HBM, read $P$ back, compute
- **B:** compute, discard, recompute later

**B** does more arithmetic, but it avoids pushing a giant tensor through HBM. On modern accelerators, that trade usually wins.

Every FlashAttention generation is built around this. FA2's goal is partly to **reduce non-matmul FLOPs,** since those don't have the same throughput as tensor-core GEMMs. It also reworks the work partitioning so more of the GPU is doing useful work at once.[^5] FA3 pushes further on Hopper, overlapping matrix multiplication, softmax, and data movement with asynchronous hardware features.[^6] FA4 responds to Blackwell, where tensor-core throughput grew faster than other resources. That shift makes exponentials and shared-memory traffic relatively more important, and FA4 is designed around it.[^7]

<!-- > The cost of a FLOP depends on what hardware executes it and what data movement surrounds it. -->

{% capture c %}
A better performance question is not merely "How many FLOPs?" but **"Which operations, on which units, with what data movement, reuse, parallelism, and synchronization?"**
{% endcapture %}
{% include callout.html type="note" title="Important Fact" content=c %}

This is the systems lesson that makes FlashAttention matter beyond attention itself. Hardware efficiency doesn't come from the math alone. It comes from co-designing the mathematical scheduling with the memory/execution hierarchy.

---

## **4. Architectural Compatibility**

### 4.1 MHA, MQA, and GQA Compatibility (Shared K/V Heads)

FlashAttention isn't tied to standard [Multi-Head Attention](https://chizkidd.github.io/2026/04/17/transformers/#self-attention--multi-head-attention) (MHA).

$$
Q \in \mathbb{R}^{B \times N\_q \times H\_q \times d}, \quad K, V \in \mathbb{R}^{B \times N\_k \times H\_{kv} \times d}.
$$

Let's look into FlashAttention's compatibility with MHA, [Multi-Query Attention](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#multi-query-attention-mqa) (MQA), and [Grouped-Query Attention](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#grouped-query-attention-gqa) (GQA). Three head-sharing regimes, distinguished by $H\_q$ versus $H\_{kv}$:

- **MHA:** Every query head has its own K/V head. $$H_q = H_{kv}$$
- **MQA:** All query heads share a single K/V head. $$H_{kv} = 1$$
- **GQA:** Several query heads share each K/V head.  $$1 < H_{kv} < H_q$$

Here's the distinction that matters:

1. MQA/GQA defines the architecture. It decides how heads share K/V projections.
2. RoPE defines the position transform. It decides how position gets baked into Q and K.
3. FlashAttention defines the execution. It decides how the resulting attention operation gets scheduled and computed efficiently.

Any combination of these three orthogonal concepts can be used together. A kernel doesn't care how many query heads share a K/V head, or whether the Q/K inputs were RoPE-rotated. It just evaluates attention between each query head and its assigned K/V head. Head sharing changes the architecture and the KV-memory footprint. RoPE changes the input representation. FlashAttention changes how the work gets scheduled on the hardware.

Current implementations do impose one shape constraint, per Dao-AILab kernel requirement. The number of query heads must be divisible by the number of KV heads[^8]:

$$
H_q \bmod H_{kv} = 0
$$

PyTorch SDPA exposes `enable_gqa`, but the docs still flag GQA support as constrained by backend and tensor shape. Production code should follow the exact version's documentation rather than assuming universal fused-kernel support.[^10]

### 4.2 Variable Lengths, Local Attention, and Dropout

Real systems aren't always the clean case: same sequence length, dense attention, no dropout.

Production implementations support a broader feature set:

- variable-length sequences
- causal attention
- sliding-window attention (SWA)
- dropout, in training-oriented interfaces
- MQA/GQA
- ALiBi-style score bias
- KV-cache decoding, including optional RoPE handling

But these are implementation capabilities, not properties of the FlashAttention mathematical idea itself.

What a given build supports depends on:

- GPU
- CUDA/ROCm backend (NVIDIA and AMD have separate implementations, with different support matrices)
- dtype
- head dimension
- mask
- library version
- kernel generation

One practical warning on dropout. PyTorch's `scaled_dot_product_attention` always applies dropout according to its `dropout_p` argument, so eval code needs to pass `0.0` explicitly when dropout should be off.[^10]

<!-- > Do not infer feature support from the name "FlashAttention." Check the exact library, kernel generation, device, dtype, head dimension, mask/bias, and training/inference path that will actually run. -->

{% capture c %}
Do not infer feature support from the name "FlashAttention." Check the exact library, kernel generation, device, dtype, head dimension, mask/bias, and training/inference path that will actually run.
{% endcapture %}
{% include callout.html type="note" title="Important Fact" content=c %}

## 5. Summary

Let's walk the whole FlashAttention story in one derivation, and hit the key things to remember.

**Ordinary attention:**

$$
O = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V.
$$

**The problem.** $QK^T \in \mathbb{R}^{N \times N}$. Materializing it creates an enormous intermediate matrix.

**Naive execution:**

$$
QK^T \to \text{store } N \times N \text{ scores} \to \text{softmax} \to \text{store } N \times N \text{ probabilities} \to \text{multiply by } V
$$

This leads to huge memory traffic.

**FlashAttention idea.** Tiling. Split $Q \to Q\_i$ and $K, V \to K\_j, V\_j$, then calculate the score tile:

$$
S\_{ij} = \frac{Q\_i K\_j^T}{\sqrt{d}}.
$$

**The problem with tiling.** Softmax needs the maximum and the denominator over the whole row.

**The solution: online softmax.**

1. Maintain $(m, \ell, a)$ where:
   - $m$ = running maximum
   - $\ell$ = running normalized exponential sum
   - $a$ = running unnormalized value-weighted sum (accumulator)

2. When a new block has a larger maximum, calculate a rescaling factor:

$$
\alpha = e^{m - m'},
$$

and rescale the old state.

Therefore, we can process the full FlashAttention pipeline without ever materializing the $N \times N$ matrix:

$$
\text{tile} \to \text{score} \to \text{online softmax} \to V \text{ accumulation} \to \text{discard tile}.
$$

**Result.** The same exact dense attention mathematics, but much better memory behavior: less HBM traffic, much smaller intermediates. Arithmetic remains $O(N^2 d)$.

**FlashAttention in one table:**

| Property | Complexity / Formulation |
|---|---|
| Memory complexity | $O(Nd)$ auxiliary attention storage |
| Compute complexity | $O(N^2 d)$ dense attention compute |
| Attention formulation | Exact dense softmax attention |

{% capture c %}
**Same exact dense attention mathematics, but much better memory behavior.** Less HBM traffic, much smaller intermediates, arithmetic unchanged at $O(N^2 d)$.<br><br>
{% endcapture %}
{% include callout.html type="note" title="The one-line summary" content=c %}

---

## **Appendix**

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

### Reference Map

| Question | Best starting source |
|---|---|
| What is the original IO-aware algorithm? | FlashAttention-1 paper [^4] |
| Why does online normalization work? | Milakov & Gimelshein [^2] |
| Was exact subquadratic-memory attention known earlier? | Rabe & Staats [^3] |
| What specifically changed in FA2? | FlashAttention-2 paper [^5] |
| What is Hopper-specific in FA3? | FlashAttention-3 paper [^6] |
| What changes on Blackwell in FA4? | FlashAttention-4 paper [^7] |
| What does the current official package support? | Dao-AILab repository [^8] |
| How does current PyTorch select SDPA kernels? | PyTorch SDPA and `sdpa_kernel` docs [^10] $^,$ [^11] |
| Why can exact backends differ numerically? | PyTorch reproducibility notes [^13] |
| How is PagedAttention different? | Kwon et al. [^14] |

{% capture c %}
Implementation support changes faster than the mathematics. For deployed systems, treat the installed framework's documentation and the exact kernel repository/release as authoritative for supported devices, dtypes, head dimensions, masks, dropout, and GQA behavior.
{% endcapture %}
{% include callout.html type="note" title="A note on how to use this map" content=c %}

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

[^15]: [Understanding FlashAttention: How IO-Aware Attention Makes Transformers Faster Without Approximating Attention](https://drive.google.com/file/d/1CLyK-9Cflcvi3fRl3qAHyzYvwJFjCyVg/view). Google Drive PDF. Accessed September 2026.

### **Citation**

If you found this blog post helpful, please consider citing it:

```bibtex
@article{obasi2026FlashAttentionPt1,
  title   = "FlashAttention: Pt 1",
  author  = "Obasi, Chizoba",
  journal = "chizkidd.github.io",
  year    = "2026",
  month   = "Sep",
  url     = "https://chizkidd.github.io/2026/09/11/understanding-flashattention/"
}
```
