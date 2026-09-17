---
layout: post
comments: true
title: "FlashAttention: Part 2"
excerpt: "Notes on the FlashAttention evolution: FA2 parallelism, FA3 on Hopper, FA4 on Blackwell, PyTorch integration, and the FA1-to-FA4 systems story."
date: 2026-09-17
mathjax: true
---

---

 _This is Part 2 of a two-part FlashAttention series. If you haven't read [Part 1](https://chizkidd.github.io/2026/09/13/flashattention/) yet, start there. Part 1 covers the fundamental memory problem, the mathematical trick, and the FA1 implementation._

---

**From FA1 to FA4: The Evolution of FlashAttention**

The [handbook](https://drive.google.com/file/d/1CLyK-9Cflcvi3fRl3qAHyzYvwJFjCyVg/view) of reference for this blogpost was inspired by this [tweet](https://x.com/techNmak/status/2098057360908685358) and is titled: _Understanding FlashAttention: How IO-Aware Attention Makes Transformers Faster Without Approximating Attention._[^15]

In Part 1, we covered the fundamental problem FlashAttention-1 (FA1) addresses, the mathematical tricks utilised, the GPU implementation of these math tricks, and the architectural compatibility ([MHA](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#multi-head-attention)/[MQA](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#multi-query-attention-mqa)/[GQA](https://chizkidd.github.io/2026/08/05/attention-efficient-scalable/#grouped-query-attention-gqa)) of FlashAttention.

In this blog post, we continue the story through the evolution of the FlashAttention family (FA2, FA3, FA4), its relationship to other efficient attention techniques (PagedAttention, sparse, linear), the training vs. inference distinction, current PyTorch integration, and the practical mental model to walk away with.

---

## **Table of Contents**

[1. FlashAttention Evolution](#1-flashattention-evolution)
- [1.1 FlashAttention-2: what changed](#11-flashattention-2-what-changed)
- [1.2 FA2 parallelism across sequence tiles](#12-fa2-parallelism-across-sequence-tiles)
- [1.3 FA2 work partitioning and non-matmul FLOPs](#13-fa2-work-partitioning-and-non-matmul-flops)
- [1.4 FlashAttention-3: the Hopper generation](#14-flashattention-3-the-hopper-generation)
- [1.5 FA3 asynchrony: overlap data movement, GEMM, and softmax](#15-fa3-asynchrony-overlap-data-movement-gemm-and-softmax)
- [1.6 FA3 FP8: performance without pretending precision is free](#16-fa3-fp8-performance-without-pretending-precision-is-free)
- [1.7 FlashAttention-4: the Blackwell generation](#17-flashattention-4-the-blackwell-generation)
- [1.8 FA4 and asymmetric hardware scaling](#18-fa4-and-asymmetric-hardware-scaling)
- [1.9 FA4 implementation and current status](#19-fa4-implementation-and-current-status)
- [1.10 FlashAttention-1 through -4 compared](#110-flashattention-1-through-4-compared)

[2. Using FlashAttention in Frameworks](#2-using-flashattention-in-frameworks)
- [2.1 PyTorch scaled-dot-product attention today](#21-pytorch-scaled-dot-product-attention-today)
- [2.2 Exactness is not bitwise identity](#22-exactness-is-not-bitwise-identity)

[3. FlashAttention vs. Other Techniques](#3-flashattention-vs-other-techniques)
- [3.1 FlashAttention vs. PagedAttention](#31-flashattention-vs-pagedattention)
- [3.2 FlashAttention vs. sparse and linear attention](#32-flashattention-vs-sparse-and-linear-attention)

[4. Training vs. Inference](#4-training-vs-inference)
- [4.1 Training, prefill, and decode are different regimes](#41-training-prefill-and-decode-are-different-regimes)

[5. Practical Engineering](#5-practical-engineering)
- [5.1 Common implementation mistakes](#51-common-implementation-mistakes)
- [5.2 Common misconceptions and the practical mental model](#52-common-misconceptions-and-the-practical-mental-model)

[6. Conclusion](#6-conclusion)
- [6.1 The mental model to have](#61-the-mental-model-to-have)
- [6.2 Summary](#62-summary)

## **Appendix**
- [Reference map: what to read for which question](#reference-map-what-to-read-for-which-question)
- [References](#references)
- [Citation](#citation)

---

## **1. FlashAttention Evolution**

### 1.1 FlashAttention-2: What Changed

FA2 didn't change the goal. It still computes exact attention without materializing quadratic intermediates. What changed was how well it uses the GPU.

The FA2 paper identifies three main changes:

1. Reduce the number of non-matmul FLOPs.
2. Parallelize attention across sequence tiles, even for a single head, to improve occupancy.
3. Repartition work among warps within a thread block to reduce shared-memory communication.

On A100, these changes produced roughly 2x speedup over FA1 in the paper's experiments, reaching 50-73% of theoretical maximum FLOPs/s. End-to-end GPT-style training reached up to 225 TFLOPs/s under the reported setup.

Those numbers aren't universal speed guarantees. They depend on GPU, shapes, dtype, causal mode, software version, and benchmark methodology.

The bigger picture: FA1's breakthrough was making exact attention IO-aware. FA2's contribution was better parallelism and work partitioning around the same algorithmic idea. Every generation since has followed the same pattern. Keep the semantics, adapt the kernel pipeline to whatever the hardware provides.

My breakdown of what each generation was actually doing:

- **FA1:** make exact attention IO-aware.
- **FA2:** reduce non-matmul FLOPs, parallelize across sequence tiles, improve warp-level work partitioning.

The question FA2 is really asking: *"How can we utilise the GPU more effectively?"*

The 2x speedup over FA1 on A100 is real, but it's a benchmark-specific, GPU-specific number. Not a universal guarantee.

{% capture c %}
**Important wording.** FA1's breakthrough was IO-aware tiled exact attention. FA2's breakthrough was better parallelism and work partitioning around the same high-level algorithmic idea.

This pattern will repeat: later generations keep the semantics but adapt the kernel pipeline to the resources and bottlenecks of newer GPU architectures.
{% endcapture %}
{% include callout.html type="note" title="The pattern across generations" content=c %}

### 1.2 FA2 Parallelism Across Sequence Tiles

A GPU needs enough independent work to keep its streaming multiprocessors busy. In FA1, the natural unit of parallelism was roughly batch times head. If batch size and head count were small, that could leave the hardware under-occupied even when the sequence itself was long.

FA2 adds parallelism along the sequence dimension. Different thread blocks can work on different query-row tiles for the same attention head. Query rows are independent until their own key/value reductions are complete, so this exposes more parallel work without changing attention semantics.

A simplified mental picture: query rows get split into blocks (block 0, block 1, ..., block 5), and each block gets assigned to a CTA. A CTA is a Cooperative Thread Array, a logical grouping of threads that execute together on a single SM.

The important point isn't the exact block assignment, which is kernel-specific. It's that long sequence length itself becomes a source of parallelism.

The practical scenario where this matters:

- batch size is small,
- number of heads is small,
- sequence length is huge.

In that case FA1 might not expose enough independent work. A GPU has many processing units. If you only parallelize over batch times heads, some of those units sit idle. FA2 adds parallelism along the sequence dimension, so different query-row tiles can be processed independently. That gives the GPU more work to schedule.

Key insight: **long sequence length itself can become a source of parallelism.**

More parallel blocks doesn't automatically mean better. You still have constraints: occupancy, registers, shared memory, tile size, data reuse. Kernel design is an optimization problem over hardware resources, not just a request for maximum thread count.

{% capture c %}
**More parallel blocks are not automatically better.** Tile size, occupancy, registers, shared memory, and data reuse interact. Kernel design is an optimization problem over hardware resources, not just a request for maximum thread count.
{% endcapture %}
{% include callout.html type="note" title="Optimization is not just thread count" content=c %}

### 1.3 FA2 Work Partitioning and Non-Matmul FLOPs

FA2 also changes how work inside each thread block is divided.

In FA1, warp partitioning could require partial results to be written to shared memory and then combined. That creates communication overhead. Picture it: warp writes partial result to shared memory, another warp reads it, combines it, writes again. Every round trip through shared memory costs cycles.

FA2 repartitions the work so warps cooperate in a way that cuts those shared-memory reads and writes.

It also reduces operations that are expensive relative to tensor-core matrix multiplication. This includes algebraic restructuring around softmax rescaling and backward computations so fewer non-matmul FLOPs are performed.

Why does this matter? Because "one FLOP" isn't a useful performance currency across all operation types. Tensor cores execute matrix multiply-accumulate at enormous throughput. Exponentials, scalar reductions, conversions, and shared-memory synchronization all have very different limits. A tensor-core matmul and an exponential aren't equivalent from a hardware-throughput perspective.

This leads to a useful optimization principle: **after fixing one bottleneck, another bottleneck emerges.**

- FA1 attacked HBM traffic.
- FA2 exposed the following as the next important bottlenecks: occupancy, warp communication, non-matmul operations.

FA2 also refines what's saved for backward, keeping compact row-wise log-sum-exp information and recomputing tiles instead of materializing probabilities. That preserves the linear-memory character while improving the kernel's work distribution.

{% capture c %}
**FA2 is a reminder that after IO is improved, the next bottleneck may be occupancy, warp communication, or non-matmul work.** Optimization moves the bottleneck rather than making hardware constraints disappear.
{% endcapture %}
{% include callout.html type="note" title="Bottlenecks shift" content=c %}

### 1.4 FlashAttention-3: The Hopper Generation

Hopper GPUs introduced hardware features that changed the best way to schedule attention. FA3 was designed specifically to exploit those capabilities.

FA2 achieved only about 35% utilization on H100 in its measurements. That leaves a lot of headroom. FA3 attacks that gap with three broad ideas:

1. Overlap computation and data movement using asynchrony and warp specialization.
2. Interleave matrix multiplication and softmax so different execution units stay busy.
3. Exploit FP8 hardware with quantization techniques designed to control numerical error.

On H100, the NeurIPS 2024 publication reports 1.5 to 2.0x speedup over FA2 in its benchmark suite. BF16 throughput reaches up to 840 TFLOPs/s (85% utilization), and FP8 reaches 1.3 PFLOPs/s. Again, these are hardware- and benchmark-specific empirical results, not promises for arbitrary models.

Why FA3 exists at all: GPU hardware changed. FA2 didn't fully exploit what H100 could do, based on the reported measurements (around 35% utilization).

FA3 introduces:

1. Asynchronous computation and data movement.
2. Warp specialization.
3. Overlapping GEMM and softmax.
4. FP8 support.

The important point: **the mathematical algorithm didn't suddenly change.**

FA3's BF16 path still preserves the exact-attention goal. The FP8 path intentionally introduces lower-precision arithmetic, so it needs its own numerical-accuracy discussion. That's a different mode of operation, not a different algorithm.

{% capture c %}
**FA3 is not "a more approximate FlashAttention."** The FP16/BF16 path preserves the exact-attention algorithmic goal. The FP8 path intentionally introduces lower-precision arithmetic and therefore needs its own numerical-accuracy discussion.
{% endcapture %}
{% include callout.html type="note" title="Exactness vs. precision" content=c %}

In FA3, what changed was the execution pipeline, tuned to exploit Hopper hardware.

### 1.5 FA3 Asynchrony: Overlap Data Movement, GEMM, and Softmax

Picture a naive pipeline:

```
load → GEMM → softmax → PV GEMM → load next tile
```

Each stage waits for the previous one. That leaves periods where some hardware resources sit idle while others work. Tensor cores go quiet during softmax. Memory buses go quiet during GEMM.

FA3 builds a more overlapping pipeline:

```
load next tile || QK^T GEMM || softmax/update || PV GEMM
```

The `||` means the stages are intentionally overlapped where dependencies permit. They're not mathematically independent, but the hardware can be kept busy across stage boundaries.

Two Hopper capabilities make this possible:

- **Tensor Memory Accelerator (TMA)**, which moves data asynchronously.
- **Asynchronous warp-group matrix-multiply-accumulate (WGMMA)**, which lets tensor-core work proceed without blocking the issuing warp.

FA3 also uses **warp specialization**. Different warps take responsibility for different stages of the pipeline: some handle data movement, some issue GEMMs, some handle softmax and update. That way a warp that's waiting on a memory transfer doesn't stall the whole block.

FA3 also uses a **ping-pong style schedule** to interleave block matrix multiplication and softmax. The point is to avoid leaving tensor cores idle while scalar and special-function work is performed, and vice versa.

The online-softmax recurrence itself hasn't changed. The normalization and value-accumulation stages still do what Part 1 describes. What changed is the hardware pipeline used to execute them.

This is a good moment to look at the whole progression:

- **FA1:** how do we avoid HBM traffic?
- **FA2:** how do we partition the work better?
- **FA3:** how do we overlap distinct execution resources on Hopper?

Each question follows from the previous answer exposing a new bottleneck.

{% capture c %}
FA1 asks "How do we avoid HBM traffic?" FA2 asks "How do we partition the work better?" FA3 asks "How do we overlap distinct execution resources on Hopper?"

The online-softmax recurrence is still present conceptually. What changes is the hardware pipeline used to execute the score, normalization, and value-accumulation stages.
{% endcapture %}
{% include callout.html type="note" title="Three questions, three generations" content=c %}

### 1.6 FA3 FP8: Performance Without Pretending Precision Is Free

Hopper gives you very high tensor-core throughput for FP8. That's tempting, but it isn't free. Simply converting every attention operand to FP8 produces unacceptable numerical error, because attention combines dot products, exponentials, normalization, and value accumulation over dynamic ranges that vary by tile.

FA3 introduces an FP8 path with two techniques to control the error:

- **Block quantization**, which scales values relative to their local tile rather than globally.
- **Incoherent processing**, which spreads out the effective range of the values being quantized.

In the final NeurIPS 2024 publication, FA3 reaches 1.3 PFLOPs/s on H100 with this path, and reports **2.6x lower numerical error** than the baseline FP8 attention method used for comparison.

Now the precision question. Attention contains a chain of operations:

$$
QK^T \rightarrow e^{x} \rightarrow \frac{e^{x}}{\sum e^{x}} \rightarrow PV.
$$

These operations have different numerical sensitivities. Dot products tolerate some loss. The exponential is much more sensitive. Normalization involves division. The value accumulation sums over long sequences. Each stage has its own error profile.

So low precision doesn't automatically buy you free performance. It shifts error into specific parts of the pipeline, and you have to design around that shift.

This is where two meanings of "exact" have to be kept separate.

- **Algorithmic exactness.** The algorithm still represents $\mathrm{softmax}(QK^T)V$. No pairs are skipped, no low-rank approximation is made.
- **Numerical precision.** The actual calculation may use FP8, and therefore experiences quantization and rounding error.

An algorithm can be exact in its mathematical formulation while a particular low-precision implementation of that algorithm is numerically less accurate. Both things are true at once, and neither cancels the other.

A model's acceptable precision depends on the full training and inference setup. FP8 support is a hardware-and-numerics feature layered onto the FlashAttention execution strategy, not proof that FP8 is universally interchangeable with BF16 or FP16.

{% capture c %}
**Do not conflate two meanings of "exact."** FlashAttention's tiling and online-softmax algorithm can be exact with respect to the dense attention formula, while performing that formula in a lower-precision numeric format still introduces quantization and rounding error.

The official repository continues to separate FA3's Hopper-specific implementation from other backends, which is another reason not to state FA3 capabilities as generic properties of "FlashAttention."
{% endcapture %}
{% include callout.html type="note" title="Algorithmic exactness vs. numerical precision" content=c %}

### 1.7 FlashAttention-4: The Blackwell Generation

FA4 continues the same pattern: hardware-aware redesign, this time for NVIDIA Blackwell.

Blackwell exhibits **asymmetric hardware scaling**. Tensor-core throughput increased substantially relative to Hopper, but several other resources didn't scale at the same rate. Shared-memory bandwidth and exponential throughput, in particular, lag behind.

This changes which stage is the bottleneck. Imagine a bar chart with tensor-core throughput on one side and softmax/exponential throughput on the other:

- On Hopper, tensor cores were fast, but the gap to softmax and shared-memory bandwidth was narrower.
- On Blackwell, the matrix-multiplication bar gets much taller, and the softmax/exponential bar stays roughly the same length.

When one subsystem becomes much faster than the others, operations that used to be secondary can become the bottleneck. Matrix multiplication is now extremely fast. The previously secondary softmax work is now comparatively expensive.

FA4 redesigns both the forward and backward pipelines rather than just reusing the Hopper schedule. The paper's major techniques:

- Fully asynchronous MMA pipelines and larger tiles.
- Software-emulated exponential plus conditional online-softmax rescaling in the forward pass.
- Tensor Memory and 2-CTA MMA techniques to reduce shared-memory traffic and atomic additions in the backward pass.

On B200 with BF16, the paper reports up to 1.3x speedup over cuDNN 9.13 and 2.7x over its Triton comparison, reaching 1613 TFLOPs/s (71% utilization) under the evaluated configurations.

None of this changes the big-O complexity of dense attention. FA4 attacks the new bottlenecks created by a GPU generation whose compute, memory, and function-unit balance differs from Hopper's.

{% capture c %}
FA4 does not change the big-O arithmetic of dense attention. It attacks the new bottlenecks created by a GPU generation whose compute, memory, and function-unit balance differs from Hopper.
{% endcapture %}
{% include callout.html type="note" title="Same algorithm, different balance" content=c %}

### 1.8 FA4 and Asymmetric Hardware Scaling

The FlashAttention lineage is a case study in why kernels can't be optimized once and assumed optimal forever. Let's look at two hypothetical GPU generations:

| Unit | GPU A | GPU B |
|---|---|---|
| GEMM | 100 | 200 |
| Softmax | 50 | 50 |
| Memory | 50 | 50 |

GPU B makes GEMM 2x faster. But the overall system doesn't become 2x faster automatically. Now that GEMM is less of a bottleneck, the relative cost of everything else goes up. Softmax and memory traffic, which were already on the critical path, are now even more clearly on it.

This is **asymmetric scaling.** The ratio between subsystems changes, and the optimal algorithm changes with it.

On Blackwell, matrix multiplication became fast enough that the forward pass can be constrained by softmax and exponential work rather than by GEMM. FA4 responds by using a software-emulated exponential and by skipping the online-softmax rescaling step when the running maximum doesn't actually require it. Both moves reduce pressure on the now-relatively-slower function units.

The backward pass has a different bottleneck. FA4 uses Blackwell's Tensor Memory (TMEM) and 2-CTA MMA mode to reduce shared-memory traffic and atomic accumulation overhead.

The underlying principle is general:

> A kernel is balanced against a particular hardware ratio. If tensor-core throughput improves faster than memory bandwidth or special-function throughput, the optimal algorithmic pipeline can change even when the mathematical function is identical.

This is why "FlashAttention-4 is just a faster FA3" misses the interesting part. The algorithm and the kernel pipeline are co-designed around the new asymmetries. The FA4 paper title makes that explicit: *Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling.*

It also explains why performance numbers should always name the GPU generation. A speedup on B200 is not evidence for the same ratio on H100, A100, or a non-NVIDIA accelerator. The whole point is that the ratios change.

Asymmetric scaling is one of the major systems principles behind the FA1 → FA4 evolution.

{% capture c %}
**A kernel is balanced against a particular hardware ratio.** If tensor-core throughput improves faster than memory bandwidth or special-function throughput, the optimal algorithmic pipeline can change even when the mathematical function is identical.
{% endcapture %}
{% include callout.html type="note" title="The systems principle" content=c %}

### 1.9 FA4 Implementation and Current Status

The current official Dao-AILab repository exposes FlashAttention-4 as a CuTeDSL implementation, optimized for Hopper and Blackwell GPUs such as H100 and B200.

CuTeDSL stands for **CUDA Tensor Domain Specific Language**. It's a Python-based programming framework developed by NVIDIA for writing highly optimized, low-level, SOTA GPU kernels.

One engineering motivation behind the FA4 choice is **compile-time productivity**. The FA4 paper reports substantially faster compilation, roughly 20 to 30 times, compared with the traditional C++ template-based approach used in its comparison, while still retaining the required low-level expressivity.

As of September 2026, PyPI classifies `flash-attn-4` as:

```
Development Status :: 3 - Alpha
```

That's the caveat. Current implementation status is not the same thing as an evergreen property of FlashAttention. Packages, GPU support, CUDA compatibility, and framework integration all evolve quickly.

The durable lesson is the algorithmic progression. The current library support layer changes much faster than the underlying ideas. Algorithmic knowledge evolves slowly. Library support can change overnight.

When deploying FlashAttention, it's worth separating the two. The algorithm tells you what should be possible. The current library tells you what's actually available on your specific GPU, CUDA version, and framework.

{% capture c %}
Treat this as a current implementation snapshot, not an evergreen property. "FA4 exists in the official repository" and "every production environment should replace FA2 with FA4" are very different claims.

Hardware, CUDA, PyTorch, feature, and packaging requirements evolve quickly. The durable lesson is the algorithmic progression, while exact installation and support claims should always be checked against the current official repository before deployment.
{% endcapture %}
{% include callout.html type="note" title="Implementation snapshot" content=c %}

### 1.10 FlashAttention-1 Through -4 Compared

Here's the whole evolution on one page:

| Version | Main Problem | Main Solution |
|---|---|---|
| **FA1** | Excessive memory traffic | Tiling + online softmax + fused exact attention + IO awareness |
| **FA2** | GPU underutilization | More sequence-level parallelism + better warp-level work partitioning + fewer non-matmul FLOPs |
| **FA3** | Hopper execution imbalance | Asynchronous TMA/WGMMA pipeline + GEMM-softmax overlap + warp specialization + FP8 path |
| **FA4** | Blackwell hardware asymmetry | Fully async MMA pipeline + larger tiles + optimized non-matmul work (software exp / conditional online-softmax rescale, TMEM, 2-CTA backward techniques) |

And here's the same content in the source handbook's format, with the numbers:

| | Main target | Core kernel / algorithmic emphasis | Selected paper-reported results |
|---|---|---|---|
| **FA1** | IO / HBM traffic | Tiling, online softmax, fused exact attention, IO analysis | NeurIPS 2022 paper reports substantial training speedups vs. then-current baselines |
| **FA2** | GPU utilization | More sequence-level parallelism, fewer non-matmul FLOPs, improved warp work partitioning | About 2x over FA1 and 50-73% theoretical max FLOPs/s on A100 in paper |
| **FA3** | Hopper | Warp specialization, asynchronous TMA/WGMMA pipeline, GEMM-softmax overlap, FP8 path | 1.5-2.0x over FA2 on H100 in paper, BF16 up to 840 TFLOPs/s |
| **FA4** | Blackwell-era asymmetry | Fully async MMA pipelines, larger tiles, software exp / conditional rescale, TMEM and 2-CTA backward techniques | Up to 1613 TFLOPs/s BF16 on B200, 1.3x vs. cuDNN 9.13 in paper |

A few things to hold onto when reading those numbers:

- The version number should not be interpreted as a sequence of different attention definitions. The family preserves the central goal of efficient attention evaluation while adapting the scheduling and numerical paths to newer hardware.
- The exact feature matrix differs across implementations. FA3 is strongly associated with Hopper-specific optimization, while the current FA4 repository targets both Hopper and Blackwell through its CuTeDSL path.
- These are four successive generations of efficient implementations of attention, not four different attention mechanisms. The mathematical target remains fundamentally the same for dense attention.

{% capture c %}
The version number should not be interpreted as a sequence of different attention definitions. The family preserves the central goal of efficient attention evaluation while adapting the scheduling and numerical paths to newer hardware.
{% endcapture %}
{% include callout.html type="note" title="Successive generations, not different mechanisms" content=c %}

---

## **2. Using FlashAttention in Frameworks**

### 2.1 PyTorch Scaled-Dot-Product Attention Today

Modern PyTorch provides a high-level API:

```python
torch.nn.functional.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
)
```

Conceptually, this function computes $\mathrm{softmax}(QK^T / \sqrt{d})V$. Internally, PyTorch selects an optimized backend based on the inputs.

The current PyTorch documentation says CUDA SDPA can automatically select optimized implementations based on the inputs. The main API documentation explicitly discusses FlashAttention-2, a memory-efficient attention implementation, and the C++ math implementation. The preferred mechanism for restricting the backend is `torch.nn.attention.sdpa_kernel`.

For example:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

If a requested fused kernel can't run for the supplied inputs, PyTorch can warn with reasons when other fallbacks are disabled. Backend eligibility depends on device, dtype, shape, mask, and other arguments.

Current `torch.nn.attention` documentation also exposes registration and activation hooks for newer FlashAttention implementations, including identifiers such as FA3 and FA4. This is a fast-moving integration surface, so use the installed PyTorch version's documentation as authority.

The practical point: calling the high-level API doesn't necessarily mean "I know exactly which kernel executed."

Backend selection and eligibility depend on:

- device
- dtype
- shape
- mask
- other arguments

PyTorch does expose mechanisms for restricting backend selection when needed. That's important for production and benchmarking. Always verify what actually ran.

{% capture c %}
Calling the high-level API does not necessarily mean "I know exactly which kernel executed."

If you need to know which backend is running, restrict it explicitly with `sdpa_kernel` or check the PyTorch version's registration APIs. In production and benchmarking, verify rather than assume.
{% endcapture %}
{% include callout.html type="note" title="Which kernel actually ran" content=c %}

### 2.2 Exactness Is Not Bitwise Identity

Suppose two implementations compute the same dense SDPA definition. You should still expect small numerical differences.

The reason is straightforward. **Floating-point addition is not associative:**

$$
(a + b) + c \neq a + (b + c)
$$

in finite precision, for some values. Tiling changes the reduction order. Fusion can change when values are rounded. Accumulation precision can differ.

Concretely, imagine implementation A sums a row in order 1, 2, 3, 4, and implementation B sums the same row in order 3, 1, 4, 2. Both compute the same mathematical function. Both return slightly different floating-point bits.

PyTorch's reproducibility documentation explicitly notes that SDPA backends can produce different results because they perform floating-point accumulation in different orders.

So the same mathematical function is not necessarily the same floating-point bits. Numerical validation should use appropriate tolerances rather than bitwise equality.

The expected tolerance depends on:

- FP32 vs. BF16/FP16/FP8
- accumulation precision
- sequence length and score distribution
- hardware and backend
- dropout and randomness

**Exact algorithm** means no deliberate mathematical approximation of dense attention. **Bitwise identical implementation** means every finite-precision operation lands on exactly the same bits. FlashAttention promises the first idea, not the second across arbitrary kernels.

Determinism is another separate question. Current libraries expose backend- and version-specific controls for deterministic behavior, which can carry performance or memory costs.

Bitwise equality shouldn't be used as the definition of exactness.

{% capture c %}
**Exact algorithm** means no deliberate mathematical approximation of dense attention. **Bitwise identical implementation** means every finite-precision operation lands on exactly the same bits. FlashAttention promises the first idea, not the second across arbitrary kernels.
{% endcapture %}
{% include callout.html type="note" title="Two definitions of exactness" content=c %}

---

## **3. FlashAttention vs. Other Techniques**

### 3.1 FlashAttention vs. PagedAttention

The names sound related, but they solve fundamentally different problems.

| | FlashAttention | PagedAttention |
|---|---|---|
| **Primary problem** | Efficient attention computation | KV-cache memory management |
| **Main setting** | Attention execution | LLM serving |
| **Mechanism** | Tiling, online softmax, fused/hardware-aware kernels | Paging-inspired KV-block allocation and mapping |
| **Main state** | Attention tiles and row-wise normalization/output state | Persistent per-request KV cache |
| **Changes dense attention formula?** | No for dense FlashAttention | No, primarily changes cache memory management |

Two different questions, really:

- **FlashAttention:** how do I efficiently compute $QK^T$, softmax, and $PV$?
- **PagedAttention:** how do I efficiently store and retrieve the growing KV cache for many serving requests?

PagedAttention was introduced with vLLM to reduce KV-cache waste from fragmentation and duplication in high-throughput serving. FlashAttention is about how an attention operation consumes $Q/K/V$ and leaves out the persistent KV state across requests. They can coexist in the same serving stack.

The current Dao-AILab repository even exposes a KV-cache-oriented FlashAttention interface with optional block tables, illustrating that kernel execution and cache paging are composable concerns rather than mutually exclusive ones.

In practice, the stack looks like this:

```
Paged KV cache → FlashAttention-style kernel → Attention output
```

A quick decision rule:

- If the problem statement contains $N^2$ score/probability intermediates, think **FlashAttention.**
- If it contains fragmented per-request KV-cache allocation, think **PagedAttention.**

{% capture c %}
If the problem statement contains "$N^2$ score/probability intermediates," think **FlashAttention.** If it contains "fragmented per-request KV-cache allocation," think **PagedAttention.**
{% endcapture %}
{% include callout.html type="note" title="Which one to reach for" content=c %}

### 3.2 FlashAttention vs. Sparse and Linear Attention

Three very different strategies often get grouped together because all three can make attention cheaper in some regime. They are not the same kind of change.

| | Dense FlashAttention | Sparse Attention | Linear Attention |
|---|---|---|---|
| **What it does** | Same dense attention, different execution | Fewer interactions | Different mathematical formulation |
| **Attention pattern** | Dense (all pairs) | Sparse or local | Rearranged via feature maps or associativity |
| **Scaling** | Still $O(N^2 d)$ | Can fall below full $N^2$ work | Can fall below full $N^2$ work |

- **Dense FlashAttention** keeps the dense softmax-attention function and changes how it is executed.
- **Sparse / local attention** deliberately computes only a subset of query-key interactions. If each query attends to only a window or selected blocks, arithmetic can fall well below full dense $N^2$ work, but the model's attention pattern has changed.
- **Linear-attention families** change the mathematical formulation, often using feature maps or associativity so attention can be rearranged without an explicit dense softmax score matrix. Their semantics and approximation/exactness properties depend on the specific method.

The phrase "memory-efficient attention" is too broad to identify a method by itself. Ask two questions instead:

1. Does it compute the same dense softmax attention?
2. Does it reduce arithmetic pairs, or mostly memory traffic/intermediate storage?

The first question separates dense FlashAttention from sparse/linear approaches. The second separates FlashAttention from methods that change the number of interactions.

The original FlashAttention paper explicitly contrasted its dense exact algorithm with approximate methods, and separately explored block-sparse FlashAttention as an approximate/sparse extension.

This is why "FlashAttention makes attention linear" is wrong. If a system shows near-linear scaling because it uses a local window or another sparse pattern, the sparsity is what changed the number of interactions. FlashAttention may still be the kernel underneath.

**In summary:**

- **FlashAttention**: same dense attention, different execution.
- **Sparse attention**: fewer interactions, different attention pattern.
- **Linear attention**: different mathematical formulation.

FlashAttention reduces memory requirements and IO, but dense attention remains quadratic.

{% capture c %}
"FlashAttention makes attention linear" is wrong. If a system shows near-linear scaling because it uses a local window or another sparse pattern, the sparsity is what changed the number of interactions. FlashAttention may still be the kernel underneath.
{% endcapture %}
{% include callout.html type="note" title="Common mix-up" content=c %}

---

## **4. Training vs. Inference**

### 4.1 Training, Prefill, and Decode Are Different Regimes

FlashAttention is most naturally powerful when both query and key sequence dimensions are substantial, as in training or long-prompt prefill. There are many query rows, so avoiding the materialized attention matrix and exploiting tiled GEMMs provides large benefits.

**Training.** Suppose $N\_q = N\_k = N$. There are many query rows. FlashAttention can exploit large GEMMs, tiling, massive parallelism, and avoid $N^2$ intermediates. This is a very favorable regime.

**Prefill.** Suppose the user sends a prompt of 8000 tokens. The model processes those tokens together. Again $N\_q \approx N\_k$, so dense attention is substantial. FlashAttention can be very useful here. Prefill is compute-bound.

**Decode.** Now suppose the model has already generated 8000 tokens and wants to generate token 8001. The new query might have $N\_q = 1$ while $N\_k = 8000$. So the computation is now $1 \times 8000$, not $8000 \times 8000$. Decode is memory-bound.

Autoregressive decode is different. For a single new token, $N\_q$ may be one while $N\_k$ is the accumulated context length. The operation reads a large KV cache but produces only one new query row per sequence. In that regime, KV-cache bandwidth and serving/batching behavior can dominate, and specialized decode kernels matter.

The official repository exposes `flash_attn_with_kvcache`, which can update and attend to the cache in one kernel and supports features such as MQA/GQA and optional RoPE handling.

The bottleneck changes. Decode is heavily interacting with the existing KV cache. The system's behavior is now driven by how fast it can read that cache, not by how much arithmetic it can do.

The key distinction in one line:

- **Prefill:** many query rows → large attention computation.
- **Decode:** one query row → large KV-cache memory access.

This does not mean FlashAttention is irrelevant to inference. Prefill is an attention-heavy dense regime, and decode can still use specialized kernels from the same implementation family. It means "FlashAttention speeds up LLM inference" is incomplete unless you say which phase and bottleneck.

FlashAttention also cannot eliminate other costs. MLP layers, communication in tensor parallelism, KV-cache capacity, model-weight bandwidth, sampling, and scheduler overhead remain separate concerns.

So when someone says "FlashAttention makes LLM inference faster," ask:

- Prefill or decode?
- Sequence length?
- Batch size?
- KV cache?
- GPU?
- Which kernel?

{% capture c %}
This does not mean FlashAttention is irrelevant to inference. Prefill is an attention-heavy dense regime, and decode can still use specialized kernels from the same implementation family. It means "FlashAttention speeds up LLM inference" is incomplete unless you say which phase and bottleneck.
{% endcapture %}
{% include callout.html type="note" title="Phase and bottleneck matter" content=c %}

---

## **5. Practical Engineering**

### 5.1 Common Implementation Mistakes

A running list of things that bite people in practice:

1. **Assuming the flash backend always ran.** High-level frameworks may fall back when dtype, device, shape, or mask is unsupported. Use backend controls and profiling when it matters.
2. **Leaving dropout on during evaluation.** PyTorch SDPA applies `dropout_p` exactly as supplied, so pass `0.0` when evaluation requires no dropout.
3. **Using the wrong mask convention.** Boolean mask semantics differ across APIs.
4. **Expecting bitwise agreement.** Fused kernels can reorder floating-point operations.
5. **Ignoring GQA head constraints.** Current interfaces require compatible/divisible query and KV head counts.
6. **Treating current head-dimension/dtype support as timeless.** These constraints change across releases and backends.
7. **Benchmarking without synchronization or warmup.** Asynchronous GPU execution can make naive wall-clock measurements meaningless.
8. **Calling a sparse or local pattern "dense FlashAttention."** Kernel and attention pattern are separate choices.
9. **Confusing prefill with decode.** The shapes and bottlenecks are different.
10. **Assuming the newest generation is always deployable.** Current FA4 package metadata is alpha, and hardware/software compatibility must be checked.

### 5.2 Common Misconceptions and the Practical Mental Model

A list of things people say that aren't quite right, and what's actually true.

1. **"FlashAttention approximates softmax."** No. Dense FlashAttention evaluates the dense softmax-attention function using tiled online normalization. Nothing is thrown away, nothing is approximated.

2. **"FlashAttention makes attention $O(N)$."** No. Dense arithmetic remains $O(N^2 d)$. What becomes linear is the large auxiliary attention-memory footprint with respect to sequence length. HBM traffic is reduced, but the underlying number of query-key interactions is unchanged.

3. **"It is just kernel fusion."** Fusion is part of the implementation story. But online-softmax tiling and IO-aware scheduling are algorithmic, not merely concatenating existing kernels. Fusion alone wouldn't produce the same memory profile or the same IO complexity bound.

4. **"It is the same as PagedAttention."** No. PagedAttention manages KV-cache allocation for serving. FlashAttention manages how the attention operation itself is executed. Different problems, different solutions.

5. **"Exact means bitwise identical."** No. Floating-point operation order can change rounding. Two exact implementations of the same function can produce slightly different bits, and that's expected.

6. **"Long context becomes free."** No. Dense pairwise arithmetic is still quadratic, and KV-cache and other model costs remain. What FlashAttention gives you is a smaller memory footprint and less HBM traffic, not a free lunch.

7. **"FA1, FA2, FA3, and FA4 are different attention architectures."** No. They are successive generations of efficient kernels and algorithms, shaped by different hardware bottlenecks. The mathematical target is fundamentally the same for dense attention.

---

## **6. Conclusion**

### 6.1 The Mental Model to Have

Think of FlashAttention as a **streaming matrix computation with exact streaming softmax**.

<!-- $$
\begin{array}{|c|c|c|c|}
\hline
 & & & \\ \hline
 & & & \\ \hline
 & & & \\ \hline
 & & & \\ \hline
\end{array}
$$ -->

<table style="margin: 1rem auto; border-collapse: collapse; font-family: inherit;">
  <tr>
    <td style="border: none;"></td>
    <td colspan="4" style="text-align: center; border: none;">K</td>
  </tr>
  <tr>
    <td style="border: none;"></td>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
  </tr>
  <tr>
    <td rowspan="3" style="padding-right: 6px; border: none; vertical-align: middle;">Q</td>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
  </tr>
  <tr>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
  </tr>
  <tr>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
    <td style="width: 22px; height: 22px; border: 1px solid #888;"></td>
  </tr>
</table>

Picture the full attention matrix, $Q$ on the $y$-axis and $K$ on the $x$-axis. It's a big grid. A naive implementation would calculate the entire grid and store it. FlashAttention does something different.

It brings one small region of the grid close to the compute units. It calculates that region, normalizes it using running statistics, immediately uses the result to accumulate the output, then discards the region. Then it moves on to the next region.

The mathematical interactions remain. The physical representation of the intermediate computation does not. That's the central insight.

The evolution from FA1 to FA4, in one line each:

- **FA1** → "Reduce expensive HBM movement."
- **FA2** → "Now use the GPU more effectively."
- **FA3** → "Now overlap communication and memory movement on Hopper."
- **FA4** → "Now redesign the pipeline for Blackwell's new hardware balance."

This evolution is why FlashAttention is much more interesting than just _"a faster attention kernel."_ It's a case study in **algorithm-hardware co-design**. The mathematical function can stay identical while the optimal execution strategy changes dramatically as the memory hierarchy, compute throughput, and specialized hardware change.

The final practical mental model, in four moves:

- keep score tiles near compute,
- carry only the row statistics needed to merge softmax blocks,
- immediately consume probabilities into the $V$ accumulation, and
- avoid sending the full $N^2$ attention matrix through HBM.

{% capture c %}
Think of FlashAttention as a streaming matrix computation with exact streaming softmax. The mathematical interactions remain, but the physical representation of the intermediate computation does not. That is the central insight.
{% endcapture %}
{% include callout.html type="note" title="The mental model" content=c %}

### 6.2 Summary

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

### Reference Map: What to Read for Which Question

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
@article{obasi2026understandingFlashAttention,
  title   = "Understanding FlashAttention: Personal Notes",
  author  = "Obasi, Chizoba",
  journal = "chizkidd.github.io",
  year    = "2026",
  month   = "Sep",
  url     = "https://chizkidd.github.io/2026/09/17/understanding-flashattention-part-2/"
}
```