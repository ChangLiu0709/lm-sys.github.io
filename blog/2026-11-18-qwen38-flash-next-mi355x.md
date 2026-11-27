---
title: "Qwen3.8-Flash-Next on AMD Instinct MI355X: serving and optimization with SGLang"
author: "Nikolai (@amd-nprotaso), Christian Gilli (@amd-cgilli), and Chang Liu (@ChangLiu0709) — AMD"
date: "November 18, 2026"
previewImg: /images/blog/qwen38-flash-next-mi355x/preview.png
type: blog
---

Qwen3.8-Flash-Next is an efficient, open-weight MoE model, previewing the next-generation Qwen4 architecture. It has gained significant attention in the open-source community due to its strong coding and agentic capabilities, ultra-sparse architecture, and innovations such as Gated DeltaNet (GDN) and Qwen Sparse Attention (QSA) for efficient long-context inference. Given these strengths, enabling and optimizing Qwen3.8-Flash-Next inference on AMD Instinct™ GPUs is important. In this blog, we demonstrate how to optimize its inference performance in SGLang on AMD GPUs. We benchmark inference performance using InferenceX AgentX, which replays realistic multi-turn agentic coding workloads and measures throughput and interactivity across different concurrency levels. Our optimized implementation delivers strong inference performance on AMD Instinct MI355X GPUs across all tested concurrency levels.

This post explores how we serve and optimize Qwen3.8-Flash-Next on AMD Instinct MI355X GPUs using SGLang. We begin with our latest InferenceX AgentX benchmark results, followed by a brief introduction of the model architecture, key kernel optimizations for MI355X, including grid sizing for 256 CUs, kernel fusion, MFMA-native layouts, register prefetching, and wave64, and the monokernel for low concurrency. All results can be reproduced using the commands provided in the [SGLang cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next#hw=mi355x&variant=default&quant=fp8&strategy=balanced&nodes=single&pleOffload=auto).

## Performance

We use InferenceX AgentX to evaluate inference performance, as InferenceX is a public benchmarking platform that enables fair comparisons across different hardware architectures using standardized workloads and configurations. AgentX provides realistic multi-turn agentic coding workloads that capture long-context requests, prefix caching, and subagent interactions. Its total throughput per GPU and interactivity metrics allow us to assess serving performance across different concurrency levels, reflecting real-world agentic workloads.

As shown in Figure 1., we ran Qwen3.8-Flash-Next on one MI355X at concurrency 1, 2, 12, 24, 48 and 80 on the SGLang ROCm image lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261117. Each point replays recorded coding-agent sessions from the InferenceX AgentX trace set for 3,600 s. Total throughput per GPU rises from 16.8k to 230.8k tok/s as concurrency grows, while per-user interactivity ranges from ~390 to ~27 tok/s/user. At low concurrency the monokernel drives interactivity; at high concurrency kernel fusion and HiCache keep throughput scaling.

![InferenceX AgentX: token throughput per chip vs. P90 interactivity](/images/blog/qwen38-flash-next-mi355x/agentx_performance.jpg)

<p style="text-align: center;"><em>Figure 1. Screenshot of InferenceX AgentX Qwen3.8-Flash-Next performance: total throughput per GPU against interactivity across concurrency 1–80, comparing NVIDIA B200, NVIDIA B300, and AMD MI355X.</em></p>

At high concurrency (24–80), the workload is dominated by long, largely shared prompts, so prefill speed and cache capacity matter as much as decode speed. Our MI355X run reaches about 126k tokens/s per chip at conc=24 and about 231k at conc=80, matching or exceeding B300 at the same P90 interactivity and staying well above B200 and our previous MI355X results. At low concurrency (1–2), decoding is bound by kernel launches, not arithmetic: one decode step took about 14 ms across more than 1,600 small kernels. Our monokernel, a persistent FlyDSL kernel that runs a full MTP verify step in 13 launches, cuts that to about 4 ms. P90 interactivity reaches about 390 tokens/s/user at one agent, where B300's curve falls off to a few thousand tokens/s per chip. We will give more details about the optimizations in the latter sections.

**NOTE:** As of November 19, 2026, when we drafted the blog, the performance submission shown in the figure depended on seven PRs that were still awaiting merge into SGLang's main branch.

## Qwen3.8-Flash-Next Model Architecture

<img src="/images/blog/qwen38-flash-next-mi355x/architecture.png" alt="Qwen3.8-Flash-Next Model Architecture" style="display:block; margin-left: auto; margin-right: auto; width: 80%;"></img>

<p style="text-align: center;"><em>Figure 2: Qwen3.8-Flash-Next model architecture</em></p>

Figure 2 shows the model main building blocks, these are:

- **Gated Delta Net.** 36 of the 48 layers use Gated Delta Net, a linear-attention layer with a fixed-size recurrent state per request, so its decode cost does not grow with context.
- **Query Sparse Attention.** The other 12 use Query Sparse Attention: each query attends to a learned selection of past positions, with 24 query heads sharing 2 KV heads of dimension 256.
- **Hyper-connections.** Instead of one residual stream, the model keeps 4 streams of width 2,560. Before every block a gated mix collapses them into one input. It is small but runs in every layer at every step.
- **Native MTP.** A built-in draft layer proposes tokens for speculative decoding, verified by the target in one forward pass.

### The checkpoint: MXFP4 experts with AMD Quark

For AMD Instinct GPUs, we quantized the Qwen3.8-Flash-Next model weights using [AMD Quark](https://github.com/amd/quark), a model optimization toolkit that supports post-training quantization for efficient inference on AMD hardware. The resulting quantized model reduces GPU memory consumption and improves inference efficiency. In our experiments, we serve [amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8).

The quantized checkpoint occupies approximately 116 GB, fitting comfortably on a single AMD Instinct MI355X GPU (288 GB HBM3E) while leaving enough memory for a large KV cache, enabling higher concurrency and more efficient inference serving. The MTP draft model is quantized using the same approach as the target model, allowing its MoE layers to benefit from the same optimized MXFP4 kernels.

## Kernel Optimization

To find where decode and prefill time went on MI355X, we captured Perfetto traces with `rocprofv3`, ranked the kernels by GPU time, and profiled the hottest ones with `rocprof-compute`. The math in these kernels was correct; what made them slow fell into four categories: a kernel shape that left most CUs idle, matrix-core (MFMA) work issued several times over, intermediate results passed between kernels through HBM, and a fused kernel that was silently skipped on ROCm. The following sections walk through one representative case of each, showing the problem, the fix, and how both look in rocprof.

### 1. Kernel Shape: QSA Sparse Decode Did Not Fit the GPU

First, we noticed that SGLang used `_sparse_gqa_chunk_prefill`, a Triton
*prefill* kernel, for QSA decode, and that GPU utilization during decode was
low. Looking at the kernel's shape, we found that it filled only a small
fraction of the CUs.

MI355X has **256 CUs**, each with room for 32 waves. The kernel launches one
workgroup of 8 waves per decode row and KV head, so each request gets only
2 workgroups. At decode, the concurrency equals the number of rows:

| Concurrency | Workgroups | CUs with work | Waves launched (of 8,192 slots) | CU Utilization | Kernel time |
|---|---:|---:|---:|---:|---:|
| 8 | 16 | **16 of 256** (6%) | 128 | **5.4%** | 69.5 µs |
| 16 | 32 | **32 of 256** (13%) | 256 | **10.8%** | 69.7 µs |

At concurrency 8, 240 of the 256 CUs sit idle for the entire 70 µs; at
concurrency 16, 224 do. Each CU that does have work holds a single workgroup,
filling 8 of its 32 wave slots. Doubling the concurrency doubles the number of
busy CUs but leaves the kernel time unchanged, because each workgroup still
walks through all of its positions alone.

#### A Prefill Kernel on a Decode-Sized Problem

At decode, each row is a single query token: 24 query heads (12 per KV head,
GQA) attend to the 2,051 selected positions of their KV head. The kernel
therefore has three dimensions: **rows** (1–128), **KV heads** (2) and
**positions** (2,051). `_sparse_gqa_chunk_prefill` runs here with a query
length of 1 and puts only the first two dimensions in the grid. At batch 8:

| Level | Before (Triton) |
|---|---|
| Grid | rows × KV heads = **16 workgroups** |
| Workgroup | **8 waves** |
| Tile | **16 rows × 32 tokens** (12 GQA heads padded to 16) |
| Loop | **65 steps** of 32 positions per workgroup |

This leads to three problems:

1. **The grid follows the problem, not the GPU.** With only 2 KV heads,
   batch 1 launches just 2 workgroups, and the kernel fills all 256 CUs only
   at batch 128.
2. **The largest dimension is a loop.** Each workgroup walks through all
   2,051 positions.
3. **The same calculations run many times.** A 16-row tile is a single MFMA
   tile, so the 8 waves of a workgroup cannot split it: **all 8 compute the
   same scores**. For every 32 positions, the workgroup issues 144 MFMAs where
   32 are needed — 4.5× too many (the same mistake we will see in the HC
   kernel).

#### Splitting the Positions Across the GPU

The FlyDSL kernel splits each row's positions into `num_splits` pieces
(computed from the number of CUs) and assigns one workgroup to each piece.
Every workgroup writes a partial result (`O`, max `m`, sum `l`), and a small
reduce kernel merges the partials the same way online softmax does.

| Level | After (FlyDSL), batch 8 |
|---|---|
| Grid | rows × KV heads × `num_splits` = 8 × 2 × 11 = **176 workgroups** |
| Workgroup | **4 waves** |
| Tile | per wave **16 tokens × 16 heads**, one 16 × 16 × 32 MFMA tile |
| Loop | 192 positions per split = **3 rounds** of 4 waves × 16 tokens |

![QSA decode work split at batch 8, before and after split-KV](/images/blog/qwen38-flash-next-mi355x/shape_qsa_split_kv.svg)

![QSA attention kernel CU utilization vs batch](/images/blog/qwen38-flash-next-mi355x/shape_qsa_batch.png)

With split-KV, CU utilization no longer tracks the batch size alone. Before
the change, it follows the workgroup count and reaches the full GPU only at
batch 128; after it, utilization is 46% at batch 8 and above 70% from batch 32
onward. At batch 1 it stays at 8%, because the 128-position minimum per split
caps the grid at 34 workgroups.

![QSA decode kernel trace at batch 16, before and after](/images/blog/qwen38-flash-next-mi355x/shape_qsa_trace.png)

The kernel trace at batch 16 shows the cost of the extra reduce kernel: only
3.5 µs. The attention kernel itself drops from 69.1 to 22.3 µs, so attention
plus reduce (25.9 µs) is still **2.7× faster than the old attention kernel on
its own**. The whole call goes from 4 kernels and 103 µs to 2 kernels and
26 µs (**4.0×**).

The other three old kernels disappear because the new kernel does their work
itself. `_sparse_gqa_chunk_prefill` is a varlen prefill kernel: it expects the
keys and values of each row to be packed back to back in a single buffer, with
one offset per row. Three kernels built that buffer:

- **Metadata helpers** (`_fa2_valid_counts`, `_fa2_prefix_sum`, 9 µs) count
  the valid selected positions in each row (0 ≤ position < sequence length)
  and turn the counts into row offsets in the packed buffer.
- **KV gather** (`_compact_kv`, 24.8 µs) looks up the KV cache slot of each
  selected position in `req_to_token` and copies those K/V rows into the
  packed buffer. At batch 16, that means writing 67 MB to HBM just so the
  attention kernel can read it back.

The FlyDSL kernel instead reads the top-k positions of its row directly, skips
the invalid ones, looks up each slot in `req_to_token`, and loads K/V straight
from the cache pool. It needs no counts, no offsets and no copy.

On InferenceX AgentX (MI355X TP1, concurrency 16, 1 hour), QSA attention fell
from 9.4% to 2.8% of decode GPU time. Interactivity p50 improved by
**12.7%** (127.4 → 143.5 tok/s/user), ITL p50 fell by **11.2%**
(7.85 → 6.97 ms), and throughput rose by 3.3% (78,957 → 81,530 tok/s/GPU).
The broader lesson is to compute the grid from the CU count rather than from
the problem, move part of the long loop into the grid, and check Grid Size and
CU Utilization at the batch sizes you actually serve, not just the largest
one.

### 2. MMA: The HC Mix Issued 4.5× Too Many MFMAs

The HC site was the next hotspot at decode. At batch 1,
`_hc_mix_persistent_kernel` alone accounted for 32 of the site's 43.5 µs, and
its instruction counts in rocprof-compute gave the problem away:
**57,600 MFMAs where the math needs only 12,800**.

#### Where the Extra MFMAs Come From

The HC site mixes four residual streams (10,240 values per row) with two
GEMVs: a **down** projection to 320 values and an **up** projection back to
10,240. At batch 1, the single row is padded to a 16-row MFMA tile. The Triton
kernel runs both GEMVs with the same **8 waves** per workgroup.

Triton splits the output of a `tl.dot` between waves in whole 16 × 16 tiles
and does not split k between them. That works for the up GEMV, but not for the
down GEMV:

```
up GEMV    16 rows × 128 outputs per step = 8 MFMA tiles → 8 waves, one tile each     ✓
down GEMV  16 rows × 32 outputs per tile  = 2 MFMA tiles → all 8 waves compute both   8 copies ✗
```

One MFMA instruction computes a 16 × 16 output tile over 32 values of k, so
the number of MFMAs each GEMV needs is (output tiles) × (k / 32):

```
down GEMV  320 outputs    = 20 tiles  × 10,240 / 32 = 320 k-steps → 6,400 MFMAs needed
up GEMV    10,240 outputs = 640 tiles ×    320 / 32 =  10 k-steps → 6,400 MFMAs needed
                                                                   12,800 in total
```

The up GEMV issues exactly the 6,400 it needs, one tile per wave. The down
GEMV issues its 6,400 once per wave, 8 times over: 51,200. Together, that is
**57,600 MFMAs instead of 12,800 — 4.5× too many**, the same mistake as in the
QSA kernel. The matrix cores have time to spare, since a batch-1 GEMV is
memory-bound, but a wave that is busy computing a copy is not loading the next
weights: the kernel reads its weights at only 0.43 TB/s on a GPU with 8 TB/s
of bandwidth.

#### One Wave per Tile

The FlyDSL path gives the down GEMV its own kernel and moves k into the grid:
**704 single-wave workgroups** (22 row tiles × 32 splits of k). Each wave
computes its own tile over its own slice of k, and a small reduce kernel sums
the 32 partial results. The up GEMV also gets its own kernel: 640 waves, each
owning one 16-column tile.

![HC down GEMV, before and after](/images/blog/qwen38-flash-next-mi355x/mma_hc_waves.svg)

![HC MFMA count before and after (rocprof-compute)](/images/blog/qwen38-flash-next-mi355x/mma_hc_mfma.png)

rocprof-compute confirms that the copies are gone. The down GEMV now issues
6,720 MFMAs: the 6,400 it needs plus 320 for the inject GEMV, which moved into
this kernel as one extra row tile. The up GEMV issues 6,400. In total, that is
**13,120 instead of 57,600**. The weights now arrive at 1.6–1.8 TB/s, and the
site drops from 43.5 to **16.0 µs** at batch 1 (2.7×), even with the extra
reduce kernel.

A useful rule of thumb follows from this case: a matrix product can keep only
as many waves busy as it has 16 × 16 output tiles, so size the waves for each
product and split k across workgroups when there is only one tile. In rocprof,
an MFMA count that is a multiple of the expected one means the waves are
computing copies. End to end (InferenceX AgentX, MI355X TP1, concurrency 24,
1 hour, compared with the average of two runs on main), P90 interactivity
improved by **14.9%** (53.9 → 62.0 tok/s/user), throughput by **5.6%**
(108,512 → 114,575 tok/s/GPU), and ITL p90 fell by 13.0% (18.56 → 16.14 ms).

### 3. Memory: GDN Prefill Moved Its Intermediates Through HBM

Prefill told a different story. At 32K tokens, a single GDN layer took 3.0 ms
across 9 Triton kernels, and the memory counters showed it moving **14.3 GB**
through HBM, even though its inputs and output total only about 1.1 GB.

| Kernel | Time | Workgroups | Read | Written |
|---|---:|---:|---:|---:|
| copy + l2norm (× 2) | 0.17 ms | 65,536 / 16,384 | 0.3 GB | 0.3 GB |
| kkᵀ | 0.17 ms | 24,576 | 0.3 GB | 0.4 GB |
| inverse | 0.49 ms | 24,576 | 0.3 GB | 0.2 GB |
| w, u | 0.37 ms | 24,576 | 1.1 GB | 0.8 GB |
| state scan h | **1.33 ms** | **384** | **6.2 GB** | 1.2 GB |
| output o | 0.50 ms | 49,152 | 2.8 GB | 0.4 GB |
| **total** | **3.0 ms** | | **11.0 GB** | **3.3 GB** |

(Times are from the kernel trace; bytes are from rocprof-compute block 17 and
exclude the two copy kernels.)

#### Nine Kernels, Each a Round Trip Through HBM

A GDN layer keeps a 128 × 128 state per value head. In prefill, the tokens are
cut into chunks of 64 (512 chunks at 32K), and the following steps run for
every chunk and head:

$$
\begin{array}{rlcl}
 & \textbf{step} & \textbf{shape} & \textbf{kernel} \\[2pt]
1. & q,\ k \leftarrow \operatorname{l2norm}(q),\ \operatorname{l2norm}(k) & & \text{copy + l2norm} \\
2. & A = \text{masked, decayed } \beta\, k k^\top & 64 \times 64 & k k^\top \\
3. & T = (I + A)^{-1} & 64 \times 64 & \text{inverse} \\
4. & w,\ u = \text{from } T,\ k,\ v & 64 \times 128 & w,\ u \\
5. & S_c = \text{new state from } S_{c-1},\ w,\ u,\ k & 128 \times 128 & \text{state scan } h \quad \longleftarrow \text{needs the previous chunk} \\
6. & o = \text{output from } q,\ k,\ S_{c-1},\ u & 64 \times 128 & \text{output } o
\end{array}
$$

Only the inputs and the output o need to touch HBM: about 0.7 GB read and
0.4 GB written per layer at 32K tokens. The Triton path, however, runs each
step as separate kernels over all chunks at once (step 1 alone takes four: two
copies and two l2norms), which creates three problems:

1. **Every intermediate goes through HBM.** Each kernel writes its entire
   result (A, T, w, u, …) and the next one reads it back: 14.3 GB of traffic
   where 1.1 GB is needed.
2. **state scan h writes the state of every chunk.** output o needs the state
   at the start of every chunk, so state scan h writes all 512 states to HBM
   (0.8 GB of its 1.2 GB written), and output o reads them back.
3. **state scan h runs on only 384 workgroups.** It has to process the chunks
   one after another, so it can split only over heads and state columns:
   384 workgroups (48 heads × 8 column slices), compared with 16,384–65,536
   for the other kernels in the table. It reaches only 59.5% CU Utilization
   and takes 1.33 ms, 44% of the layer.

#### One Fused Kernel, Intermediates on Chip

The fused FlyDSL kernel does the work of all 9 kernels for a chunk within one
workgroup and walks through the chunks in order, so the state and the
intermediates stay on chip and only o leaves it.

![GDN data flow per layer, before and after](/images/blog/qwen38-flash-next-mi355x/mem_gdn_flow.svg)

![GDN kernel trace, one layer at 32K tokens, before and after](/images/blog/qwen38-flash-next-mi355x/mem_gdn_trace.png)

In the trace, one layer drops from 9 kernels and 3.0 ms to a single kernel and
1.6 ms (**1.9×**). The fused kernel takes only 0.26 ms longer than the old
state scan on its own (1.58 vs 1.33 ms), yet it does the work of all
9 kernels:

- **Copies and l2norm** (4 kernels, 0.17 ms). q and k are slices of a single
  fused projection output, so `l2norm_fwd` first copies each one into a
  contiguous tensor and then writes a normalized copy. The fused kernel reads
  q and k in place and normalizes them in registers.
- **kkᵀ, inverse, w, u** (3 kernels, 1.03 ms). Their results for a chunk are
  produced and consumed inside the workgroup, so they never reach HBM.
- **State scan and output** (2 kernels, 1.83 ms) become the body of the loop:
  the state stays in registers, and o is computed from the state the workgroup
  already holds.

This case is a reminder to count bytes, not just FLOPs. When stages share a
serial loop, as GDN's chunks do, it pays to fuse them, keep the state and intermediates on
chip, and load ahead; in rocprof-compute, L2-Fabric bandwidth × kernel time
should then be close to what the math requires, and writes in particular
should match the output size. On InferenceX AgentX (MI355X TP1, 1 hour per
point), measured against the staged FlyDSL path, which already takes 2.5 ms
per layer instead of Triton's 3.0 ms, throughput improved by **3.6%** and
TTFT p95 fell by 4.9% at concurrency 12. At concurrency 24, TTFT p95 fell by
**8.7%** and throughput improved by 0.9%, and ITL p95 fell by 2.6–4.6% across
concurrency 12–24.

### 4. Dispatch: A dtype Check Switched Off the Fused QSA Indexer

The last case is not a slow kernel but a missing one. In the decode trace,
every QSA layer launched dozens of tiny PyTorch kernels before the indexer's
top-k: elementwise ops, gathers, RoPE, concatenations and index stores. SGLang
has two fused kernels for this step, and on CUDA they run; on ROCm, they never
did. At batch 8, the indexer prep of a single QSA layer ran **44 kernels and
took 125 µs** of GPU time, and 38 of those kernels ran on only 1 or
2 workgroups.

#### Why the Fused Path Was Skipped

Before the indexer scores the compressed keys, it prepares its inputs for every
decode row:

- **q prep:** RMSNorm and MRoPE on the index query, plus a store of the
  token's key and RoPE positions into a ring of 4 pending tokens.
- **k compress:** once the ring holds 4 tokens, their mean goes through
  RMSNorm and MRoPE and is stored in the compressed key cache.

The fused kernels read the RoPE cos/sin table as `const float*`, so
`_use_fused_prep()` enables them only when `cos_sin_cache` is fp32. On CUDA it
is. On HIP, however, three code paths in the rotary embedding converted it to
bf16:

1. **The init guard** kept fp32 only on CUDA and XPU, so on HIP the cache was
   created in bf16.
2. **`_match_cos_sin_cache_dtype`** (in `base.py` and `mrope.py`) converted it
   to the query dtype on the first forward pass.
3. **`fallback_rotary_embedding`** converted it to the query dtype as well.

As a result, the check returned False and the indexer fell back to its eager
PyTorch path. Because the results stayed correct, nothing failed and nothing
was logged; only the kernel trace revealed it.

#### Keeping the Cache in fp32

The fix makes HIP behave the way CUDA already did: the init guard keeps fp32,
and the two later conversions only move the cache to the right device without
changing its dtype. The cache is a read-only table of 256K positions ×
64 values, so keeping it in fp32 costs only about 64 MB.

![QSA indexer prep, before and after the fix](/images/blog/qwen38-flash-next-mi355x/dispatch_qsa_indexer.svg)

![QSA indexer prep kernel trace at batch 8, before and after](/images/blog/qwen38-flash-next-mi355x/dispatch_qsa_indexer_trace.png)

With the fused path back, the indexer prep of one layer drops from 45 kernels
and 131 µs to 3 kernels and 16.6 µs (**7.9×**). The q/k projection GEMM
(5.8 µs) is identical in both cases; the 44 eager kernels (125 µs) collapse
into 2 fused ones (10.7 µs):

- **q prep** (`qsa_index_q_prep_kernel`, 5.8 µs) replaces the eager q path:
  RMSNorm, the cos/sin lookup, RoPE, a concatenation of the rotated and
  unrotated halves, and the index stores into the ring.
- **k compress** (`qsa_index_k_compress_kernel`, 4.9 µs) replaces the eager
  compress path: a gather of each group's 4 keys, their mean, RMSNorm, the
  same RoPE chain, and the store into the compressed cache.

None of the eager kernels does much work: most take 2–5 µs on 1 or
2 workgroups, which is mostly launch and ramp-up overhead. Across 12 QSA
layers, that adds up to about 1.4 ms of GPU time per decode step at batch 8
(12 × 115 µs); measured in the server, the fallback cost about 2.3 ms per
decode step, roughly 20% of the step.

For such a small change, the end-to-end effect was large. On InferenceX
AgentX (MI355X TP1, concurrency 8, 1 hour), p50 interactivity improved by
**28%** (91.2 → 117.1 tok/s) and p90 by **75%** (45.3 → 79.4 tok/s), ITL p50
fell by **23%** (11.0 → 8.5 ms), TTFT p50 fell by 28% (1,020 → 739 ms), and
throughput rose by 8.5% (31,611 → 34,291 tok/s/GPU). The lesson here is less
about kernel design than about verification: a fast path behind a runtime
check can switch itself off without an error, leaving the output correct but
slow. If a layer launches far more kernels on one platform than on another,
look for a fallback — in the kernel trace, count the kernels per layer and
watch for runs of tiny kernels on one or two workgroups.

## Improving interactivity at low concurrency with monokernels

When we first benchmarked this model we quickly realized that optimizing the low concurrency region (i.e., <= 8) was going to be a challenge.
To understand why let's take a look at some traces:

![Traces before monokernel](/images/blog/qwen38-flash-next-mi355x/before_mono.png)

<p style="text-align: center;"><em>Figure 3: Kernel trace of a full decode step at concurrency 1 without monokernel</em></p>

Figure 3 shows the full trace of a single decode step at concurrency 1. As you can see, it takes roughly 14ms and over **1600 individual kernel launches**.
Now, launching a lot of kernels is not necessarily a problem as HIP graphs can get rid of kernel launch overheads but in this case it actually is. The issue is that these kernels are all very small so the GPU is not doing useful work but rather spending most if the time moving data from and to HBM memory.

In this scenario, the most practical optimization strategy is simply trying to reduce the HBM traffic as much as possible, this means fusing multiple operations together.

In order to achieve the best possible performance at low concurrency, we decided to fuse as many kernels as possible into a single (mono) kernel.
As we saw in figure 2, the basic building block for Qwen3.8-Flash-Next is 3x GDN layers followed by a QSA layer, repeated 12 times for a total of 48 hidden layers. We implemented a specialized FlyDSL kernel that replaces this basic building block with a **single kernel launch**.

![Traces with monokernel](/images/blog/qwen38-flash-next-mi355x/after_mono.png)

<p style="text-align: center;"><em>Figure 4: Kernel trace of a full decode step at concurrency 1 with the new FlyDSL monokernel</em></p>

Figure 4 shows a full decode step with the monokernel on. With this, we got rid of over 1600 kernel launches and replaced them with just **13** and went from almost **14ms to 4ms**, that's over **3x improvement** in decode speed.

For more details, the monokernel source code is available on [GitHub](https://github.com/sgl-project/sgl-amd-mono/tree/main/sgl_amd_mono/qwen38_flash_next).

## Conclusion

We then introduce the model architecture and highlight key SGLang kernel optimizations: **GDN, QSA, and hyper-connection kernels for high concurrency**, and **a persistent FlyDSL monokernel for low concurrency**. Together, these hardware-aware optimizations deliver strong inference performance across the benchmarked concurrency range.

## Acknowledgements

We thank the Qwen team, the SGLang community, and the AMD AITER, FlyDSL, Quark, ROCm and InferenceX contributors whose work this post builds on. [TODO: names.]

## Test configuration and disclaimer

**Software stack of the recipes above:** `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261117` (SGLang `c67db7c762`, AITER `581579ab`, FlyDSL 0.4.2), patched with the open PRs listed above. **Numbers in Performance and Accuracy** were measured on one GPU of an 8x AMD Instinct MI355X node in October 2026 with `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261026` (SGLang `5ef99013`, AITER `285579b6`), patched with the PRs open at the time, through the InferenceX AgentX agentic-coding harness (AIPerf replay of recorded coding-agent traces, 3,600 s per point, InferenceX run [40420614823](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40420614823)). Per-PR numbers quoted in the optimization sections come from each pull request's own benchmarks, on the base and image given there. Speculative-decoding throughput uses InferenceX's committed synthetic acceptance length; accuracy uses lm-eval with real verification. Results on other hardware, software versions, settings or workloads will differ; these figures are for illustration and are not a performance guarantee.

#### Article by

Authors: Nikolai (@amd-nprotaso), Christian Gilli (@amd-cgilli), and Chang Liu (@ChangLiu0709) — AMD.
