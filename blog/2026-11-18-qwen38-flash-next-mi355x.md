---
title: "Qwen3.8-Flash-Next on AMD Instinct MI355X: serving and optimization with SGLang"
author: "Nikolai (@amd-nprotaso), Christian Gilli (@amd-cgilli), and Chang Liu (@ChangLiu0709) — AMD"
date: "November 18, 2026"
previewImg: /images/blog/qwen38-flash-next-mi355x/preview.png
type: blog
---

Qwen3.8-Flash-Next is an efficient, open-weight MoE model, previewing the next-generation Qwen4 architecture. It has gained significant attention in the open-source community due to its strong coding and agentic capabilities, ultra-sparse architecture, and innovations such as Gated DeltaNet (GDN) and Qwen Sparse Attention (QSA) for efficient long-context inference. Given these strengths, enabling and optimizing Qwen3.8-Flash-Next inference on AMD Instinct MI355X GPUs is important. In this blog, we demonstrate how to improve its inference performance through a series of optimizations in SGLang on AMD GPUs. We benchmark inference performance using the InferenceX AgentX benchmark, which replays realistic multi-turn agentic coding workloads and measures throughput and interactivity across different concurrency levels. Our optimized MI355X implementation outperforms NVIDIA B200 and B300 GPUs across all tested concurrency levels.

This post explores how we serve and optimize Qwen3.8-Flash-Next on AMD Instinct MI355X GPUs using SGLang. We begin with our latest InferenceX AgentX benchmark results, followed by a introduction of the model architecture, key kernel optimizations for MI355X, including grid sizing for 256 CUs, kernel fusion, MFMA-native layouts, register prefetching, and wave64, and the monokernel for low concurrency. All results can be reproduced using the commands provided in the [SGLang cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next#hw=mi355x&variant=default&quant=fp8&strategy=balanced&nodes=single&pleOffload=auto).

## Performance

We use InferenceX AgentX to evaluate inference performance, as InferenceX is a public benchmarking platform that enables fair comparisons across different hardware platforms using standardized workloads and configurations. AgentX provides realistic multi-turn agentic coding workloads that capture long-context requests, prefix caching, and subagent interactions. Its throughput and interactivity metrics allow us to assess serving performance across different concurrency levels, reflecting real-world agentic workloads.

As shown in Figure 1., we ran Qwen3.8-Flash-Next on one MI355X at concurrency 1, 2, 12, 24, 48 and 80 on the SGLang ROCm image lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261117. Each point replays recorded coding-agent sessions from the InferenceX AgentX trace set for 3,600 s. Throughput per GPU rises from 16.8k to 230.8k tok/s as concurrency grows, while per-user interactivity falls from ~390 to ~27 tok/s/user. At low concurrency the monokernel drives interactivity; at high concurrency kernel fusion and HiCache keep throughput scaling.

![InferenceX AgentX: token throughput per chip vs. P90 interactivity](/images/blog/qwen38-flash-next-mi355x/agentx_performance.jpg)

*Figure 1. Total throughput per GPU against interactivity across concurrency 1–80.*

At high concurrency (24–80), the workload is dominated by long, largely shared prompts, so prefill speed and cache capacity matter as much as decode speed. Our MI355X run reaches about 126k tokens/s per chip at conc=24 and about 231k at conc=80, matching or exceeding B300 at the same P90 interactivity and staying well above B200 and our previous MI355X results. All of it runs on the public SGLang ROCm image. At low concurrency (1–2), decoding is bound by kernel launches, not arithmetic: one decode step took about 14 ms across more than 1,600 small kernels. Our monokernel, a persistent FlyDSL kernel that runs a full MTP verify step in 13 launches, cuts that to about 4 ms. P90 interactivity reaches about 390 tokens/s/user at one agent, where B300's curve falls off to a few thousand tokens/s per chip. We will give more details about the optimizations in the latter sections.

**NOTE:** Until the day Nov 19, 2026, we wrote the blog and the performance submission shown in the figure depended on 7 PRs [PR links] that awaited to be merged to the SGLang main branch.

## Qwen3.8-Flash-Next Model Architecture

The model architecture of Qwen3.8-Flash-Next is shown in Figure 2. For its inference, these properties of the model architecture matter:

- **Gated Delta Net.** 36 of the 48 layers use Gated Delta Net, a linear-attention layer with a fixed-size recurrent state per request, so its decode cost does not grow with context.
- **Query Sparse Attention.** The other 12 use Query Sparse Attention: each query attends to a learned selection of past positions, with 24 query heads sharing 2 KV heads of dimension 256.
- **Hyper-connections.** Instead of one residual stream, the model keeps 4 streams of width 2,560. Before every block a gated mix collapses them into one input. It is small but runs in every layer at every step.
- **Native MTP.** A built-in draft layer proposes tokens for speculative decoding, verified by the target in one forward pass.

![Qwen3.8-Flash-Next Model Architecture](/images/blog/qwen38-flash-next-mi355x/architecture.png)
*Figure 2: Qwen3.8-Flash-Next model architecture*

### The checkpoint: MXFP4 experts with AMD Quark

On AMD MI GPUs, we quantized the Qwen3.8 Flash Next model weights using AMD Quark, which is [one sentence of Quark].
Then [the model can be served in a quantized form which runs more efficiently and reduce the comsuption of GPU memory]. In our experiments, we serve `[amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8)`.

The whole checkpoint is about 116 GiB, so it reduced the comsuption on one MI355X (288 GB HBM3E) with room for a large KV cache, which makes ... [once sentence: serving more efficient]. The MTP draft model is also quantized like the way we quantized the target model, so the draft's MoE runs on the same MXFP4 kernels.

## Kernel Optimization

# Porting NVIDIA kernels to MI355X: what you need to rewrite

A kernel tuned for B300 usually *runs* on MI355X (gfx950). Triton compiles for
ROCm, and CUDA code can be HIPified. But it rarely runs *well*. We saw this
while bringing up Qwen3.8-Flash-Next in SGLang on MI355X. Three parts of the
model needed new kernels, and in all three the math stayed the same. What we
rewrote was how the work is split, how data reaches the matrix cores, and
where the kernel boundaries are. That gave 1.4–5.7× at the kernel level and up
to +15% end-to-end interactivity.

## Where the kernels sit in Qwen3.8-Flash-Next

Qwen3.8-Flash-Next is a hybrid model with 48 layers:

- **36 GDN layers** (Gated DeltaNet linear attention) keep a fixed-size state
  instead of a KV cache. Prefill runs a chunked scan over 64-token chunks.
- **12 QSA layers** (Qwen sparse attention, every fourth layer). An indexer
  picks 512 compressed blocks of 4 tokens plus up to 3 recent tokens, so
  each query attends to only 2,051 positions.
- **Hyper-connections (HC).** The model keeps four residual streams. Around
  every attention and MLP block, an HC site mixes the streams into the block
  input and injects the block output back.

All numbers below are for one MI355X at TP1, with the MXFP4 checkpoint and MTP.
These are the shapes the kernels see:

| Kernel | Shapes at TP1 | What limited it on MI355X |
|---|---|---|
| QSA sparse decode | 24 query heads, 2 KV heads (GQA group 12), head dim 256, top-k 2051 | Grid of rows × 2 KV heads, plus an extra gather pass |
| GDN chunked prefill | 48 value heads, 16 key heads, K = V = 128, 64-token chunks | 8 launches per layer, per-chunk state round trip through HBM |
| HC site, decode | 4 streams × 2560 = K 10240, low rank R 320, M ≤ 128 rows | Norm, mix, inject GEMV and combine as separate small kernels |

| Kernel | NVIDIA path in SGLang | Rewrite for MI355X | Kernel gain | End to end (MI355X, TP1) |
|---|---|---|---|---|
| QSA sparse decode ([sglang#45196](https://github.com/sgl-project/sglang/pull/45196), [aiter#6670](https://github.com/ROCm/aiter/pull/6670)) | Triton KV gather + FlashInfer TRT-LLM decode | One fused FlyDSL kernel with split-KV | 120 → 33 µs/call (3.4–5.7×) | +12.7% p50 interactivity, −11% ITL |
| GDN chunked prefill ([sglang#44103](https://github.com/sgl-project/sglang/pull/44103), [aiter#6496](https://github.com/ROCm/aiter/pull/6496)) | FlashInfer CuTe DSL chunk kernel | One fused kernel, state kept on chip | 1.6× vs best multi-kernel path, 2.2× vs Triton | up to +3.6% throughput, −8.7% TTFT p95 |
| HC site, decode ([sglang#48648](https://github.com/sgl-project/sglang/pull/48648)) | CuTe DSL split-K GEMM + separate norm/combine | Norm and inject fused into the mix | 3.0× at M=1, 1.4× at M=128 | +14.9% P90 interactivity, +5.6% throughput |

The code below is taken from these PRs. Lines marked `...` are left out.

## 1. Size the grid to fill all 256 CUs (QSA sparse decode)

### 1.0 The basic idea

A few terms first:

- **CU** (compute unit) is the basic execution block of the GPU. MI355X has
  **256 CUs**, many more than the GPUs the original kernels were sized for.
- **Workgroup (WG)** is the AMD name for a thread block. One WG always runs on
  one CU.
- **Grid** is the set of all WGs that one kernel launch creates. When you call
  `launch(grid=(A, B, C))`, you get A × B × C workgroups.

The kernels in this section use so much LDS or so many registers that only
**one WG fits on a CU**. So the number of busy CUs is simply
`min(grid size, 256)`:

```
grid = 16 WGs                          grid = 256 WGs
┌────────────────────────────────┐     ┌────────────────────────────────┐
│■■■■■■■■■■■■■■■■□□□□□□□□□□□□□□□□│     │■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■│
│□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□│     │■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■│
│□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□│     │■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■│
│   ...  (8 rows × 32 = 256 CUs)  │     │   ...                          │
└────────────────────────────────┘     └────────────────────────────────┘
 ■ busy CU   □ idle CU                  all 256 CUs pull data from HBM
 16 / 256 = 6% of the GPU works
```

Each kernel has two kinds of dimensions:

| Kind | Where it lives | Example | Adds parallelism? |
|---|---|---|---|
| **Grid dimension** | one WG per index | rows, heads | **yes**, more WGs |
| **Loop dimension** | a `for` loop *inside* one WG | KV positions, chunks, GEMM K | **no**, a longer loop in the same WG |

The kernel finishes when its slowest WG finishes. If the grid dimensions are
small (at decode they often are) and the loop dimension is long, a few WGs
each run a long loop, and the rest of the GPU waits.

The fix: **move part of the loop dimension into the grid**, until there are
about 256 WGs. QSA decode (1.1) needed exactly this.

### 1.1 QSA sparse decode ([sglang#45196](https://github.com/sgl-project/sglang/pull/45196))

#### What the kernel computes

At decode, each **row** is one query token. In every decode step each request
adds one row, and MTP verification adds one row per draft token. For one row
the dimensions are:

```
one row (one query token), TP1

  Q          24 query heads × 256 (head dim)
             ├─ q heads  0..11 ──► KV head 0     GQA: 12 q heads share
             └─ q heads 12..23 ──► KV head 1          one KV head

  K, V       per KV head: 2,051 selected positions × 256
             (indexer: 512 blocks × 4 tokens + up to 3 recent tokens)

  output     24 × 256

  loop work  2,051 positions × (Q·Kᵀ, softmax, P·V)   ◄── the long dimension
```

So there are three dimensions: **rows**, **KV heads** (only 2) and
**KV positions** (2,051).

#### Before: one WG per (row, KV head)

The old ROCm path was Triton. The grid covered rows and KV heads, and each WG
looped over all 2,051 positions by itself. Take the setup the 120 → 33 µs
numbers were measured on: 8 requests at 32K context, which gives 8 rows.

```
grid = rows × KV heads = 8 × 2 = 16 WGs

                 KV positions  0 ─────────────────────────────────► 2050
row 0, kv 0    [ WG  ████████████████████████████████████████████████ ]
row 0, kv 1    [ WG  ████████████████████████████████████████████████ ]
row 1, kv 0    [ WG  ████████████████████████████████████████████████ ]
   ...
row 7, kv 1    [ WG  ████████████████████████████████████████████████ ]
                     each WG reads 2,051 × 256 of K and V, serially

CUs:  16 busy, 240 idle (6%)
```

#### Why that is slow

Decode attention is memory bound: almost all the time goes into reading K and
V from HBM. One CU can only have a limited number of loads in flight. With 16
CUs working, the kernel uses a small part of the HBM bandwidth, and every WG
runs the full 2,051-position loop. At 1 row it is even worse: 1 × 2 = 2 WGs
on 256 CUs.

Qwen3.8 makes this worse than most models because it has only **2 KV heads**.
A model with 8 KV heads would get 4× more WGs from the same rows.

#### Change: split the KV positions across WGs (split-KV)

The new FlyDSL kernel cuts each row's 2,051 positions into `num_splits` pieces
and gives each piece its own WG. The grid grows from (rows, KV heads) to
**(rows, KV heads, `num_splits`)**:

```
grid = rows × KV heads × num_splits = 8 × 2 × 11 = 176 WGs

                 KV positions  0 ─────────────────────────────────► 2050
                 split 0   split 1   split 2         split 10
                 0..191    192..383  384..575  ...   1920..2050
row 0, kv 0    [  WG   ][  WG   ][  WG   ]  ...  [ WG  ]    ──► 11 partials
row 0, kv 1    [  WG   ][  WG   ][  WG   ]  ...  [ WG  ]    ──► 11 partials
   ...
row 7, kv 1    [  WG   ][  WG   ][  WG   ]  ...  [ WG  ]    ──► 11 partials

CUs:  176 busy, 80 idle (69%), and every WG's loop is 11× shorter
```

A WG now sees only part of the softmax, so it cannot write the final output.
It writes a **partial** for each of its 12 query heads:

- `O_i`: the not-yet-normalized output over its positions (256 values),
- `m_i`: the max score in its piece,
- `l_i`: the sum of `exp(score − m_i)` in its piece.

A small **reduce kernel** then merges the partials of each (row, query head),
the same way online softmax merges blocks:

```
m = max_i m_i
O = Σ_i exp(m_i − m) · O_i   /   Σ_i exp(m_i − m) · l_i
```

Inside one WG the work is split once more. 4 waves each take a 16-token tile,
so one *round* covers 64 positions. A split is always a whole number of
rounds:

```
one split = 192 positions = 3 rounds

            wave 0     wave 1     wave 2     wave 3
round 0   [ 0..15  ] [ 16..31 ] [ 32..47 ] [ 48..63  ]
round 1   [ 64..79 ] [ 80..95 ] [ 96..111] [112..127 ]
round 2   [128..143] [144..159] [160..175] [176..191 ]
```

#### How `num_splits` is chosen

It is computed from the CU count at run time, not hard-coded:

```python
# aiter/ops/flydsl/kernels/qsa/sparse_decode_attn.py
# A KV split always covers whole wave rounds of 16-token tiles.
SPLIT_TOKEN_ALIGN = NUM_WAVES * TILE_TOKENS  # 4 waves x 16 tokens = 64

# aiter/ops/flydsl/qsa_sparse_decode.py
# Measured on MI355X (Hkv in {1, 2}, batch 1-128): one CTA per CU beats two, since
# every extra split adds a prologue and a partial to merge; and below two 16-token
# tiles per wave the fixed per-CTA latency outweighs the extra parallelism.
_CTAS_PER_CU = 1
_MIN_TOKENS_PER_SPLIT = 2 * SPLIT_TOKEN_ALIGN

def _split_layout(topk: int, num_splits: int) -> tuple[int, int]:
    """Tokens per split (whole wave rounds) and the split count that covers topk."""
    per_split = -(-topk // max(1, num_splits))
    per_split = -(-per_split // SPLIT_TOKEN_ALIGN) * SPLIT_TOKEN_ALIGN
    return per_split, -(-topk // per_split)

def recommended_num_splits(
    num_rows: int, num_kv_heads: int, topk: int, device: torch.device
) -> int:
    """KV splits that fill the GPU at this batch; fixed per CUDA-graph batch size."""
    _, num_cus = _device_info(torch.device(device).index or 0)
    # (1) enough splits for one workgroup per CU
    wanted = -(-(_CTAS_PER_CU * num_cus) // max(1, num_rows * num_kv_heads))
    # (2) but no split shorter than 128 tokens
    wanted = min(wanted, max(1, -(-topk // _MIN_TOKENS_PER_SPLIT)))
    # (3) round each split up to whole 64-token rounds
    return _split_layout(topk, wanted)[1]

# the kernel launch: one workgroup per (row, KV head, split)
sparse_decode_attn_kernel(...).launch(
    grid=(num_rows, num_kv_heads, num_splits),
    block=(BLOCK_THREADS, 1, 1),
    stream=stream,
)
```

(`-(-a // b)` is ceiling division.) Each of the three steps has a reason:

| Step | Rule | Why |
|---|---|---|
| (1) | aim for 1 WG per CU | More WGs than CUs does not help. Each extra split adds its own prologue and one more partial to merge. |
| (2) | at least 128 positions per split | That is 2 tiles per wave. With less, the fixed cost of starting a WG is larger than the gain. |
| (3) | round up to 64 positions | One round is 4 waves × 16 tokens. A partial round would leave waves idle. |

Worked example, 8 rows, top-k 2051:

1. 256 CUs / (8 rows × 2 KV heads) = 16 splits would give one WG per CU.
2. The 128-position minimum allows up to ⌈2051 / 128⌉ = 17 splits, so 16 stays.
3. 2051 / 16 = 129 positions per split, rounded up to 192 (3 × 64). Covering
   2,051 positions with 192-position splits takes ⌈2051 / 192⌉ = 11 splits.

The same rule at other batch sizes (TP1, 2 KV heads):

| Rows | Old grid (rows × 2) | `num_splits` | Positions per split | New grid | CUs busy |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 17 | 128 | 34 | 13% (limited by step 2) |
| 8 | 16 | 11 | 192 | 176 | 69% |
| 16 | 32 | 7 | 320 | 224 | 88% |
| 32 | 64 | 4 | 576 | 256 | 100% |
| 128 | 256 | 1 | 2,051 | 256 | 100%, no split, no reduce |

At 128 rows the old grid already fills the GPU, so nothing is split and the
reduce kernel is skipped. The split count depends only on the batch size, so
it is fixed for each captured graph batch size (see section 7).

#### Result

Time per call, 8 rows at 32K context:

```
before (Triton)                                              120 µs
├─ attention  (_sparse_gqa_chunk_prefill, 16 WGs)  ███████████████████  71.8
├─ KV gather  (_compact_kv)                         ██████████          39.4
└─ helpers    (_fa2_valid_counts, _fa2_prefix_sum)  ██                   8.8

after (FlyDSL)                                                33 µs
├─ attention  (sparse_decode_attn_kernel, 176 WGs)  ███████             28.9
└─ reduce     (sparse_decode_reduce_kernel)         █                    4.4
```

Split-KV is the reason the attention part dropped from 71.8 to 28.9 µs. The
gather and helper kernels went away for a different reason (fusion, see
section 2).

The speedup is largest where the old grid was emptiest:

| Batch (TP1) | Old grid | FlyDSL | Triton | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 2 WGs | 16.6 µs | 89.6 µs | 5.4× |
| 16 | 32 WGs | 25.3 µs | 112.4 µs | 4.4× |
| 128 | 256 WGs | 100.9 µs | 344.4 µs | 3.4× |

At batch 128 the grid was already full. The 3.4× there comes from the fusion
and the MFMA rewrite (sections 2 and 3), not from the split.

### 1.2 Summary of section 1

```
long dimension     2,051 KV positions
can it be split?   yes: softmax partials merge in a small reduce
before             grid = rows × KV heads (16), long loop per WG
change             move positions INTO the grid (rows × KV heads × splits)
knob               num_splits from the CU count
```

Compute the grid from the CU count at run time. Do not copy the grid from the
original kernel (it was sized for a GPU with far fewer compute units), and do
not hard-code it.

## 2. Rewrite the MMA part for MFMA, don't translate it

### 2.0 The basic idea: two different matrix engines

The original kernels and MI355X both use matrix units, but in very different
ways.

```
original design: large-tile MMA + TMEM      MI355X (gfx950), MFMA

  one thread issues one MMA                   every wave (64 lanes) issues
  for the whole thread block                  its own MFMA

  tile: 128 × N per block                     tile: 16 × 16 (or 32 × 32) per wave
        (256 × N with a block pair)

  operands:  shared memory (TMA puts          operands:  registers, every lane
             them there)                                 holds a small piece

  accumulator: Tensor Memory (TMEM),          accumulator: registers, every lane
             256 KB per core, NOT registers              holds a small piece
```

What this means for a kernel:

| | Original design | MI355X | Consequence |
|---|---|---|---|
| Who holds the data | the hardware (SMEM, TMEM) | each of the 64 lanes | You must know which lane holds which element |
| Tile size | large (128 × N) | small (16 × 16) | A big tile is many MFMAs, and you choose how to place them |
| Accumulator | TMEM, outside the register file | registers | A big accumulator takes registers away from everything else |

The register cost is real. Take a 128 × 256 fp32 accumulator, a normal size
when the kernel was designed around TMEM. On MI355X, spread over a workgroup of 4 waves (256 lanes), it takes
128 × 256 / 256 = **128 registers per lane**, before any operands, addresses
or prefetch buffers. In the original design the same accumulator sits in
TMEM and costs no registers.

So a tile layout or fragment mapping copied from the original kernel is wrong (the
lanes hold other elements) or slow (it does not fit in registers, or it needs
extra copies through LDS). You have to design the MMA part again. The rest of
this section shows how we did it for three cases:

| | Kernel | Problem | Rewrite |
|---|---|---|---|
| 3.1 | QSA sparse decode | chain QKᵀ → softmax → PV without going through LDS | transpose the problem so the accumulator is the next operand |
| 3.2 | GDN chunked prefill | triangular inverse is a serial loop | compute it with matrix products only |
| 3.3 | HC GEMV | each lane loads its own fragment from HBM | store the weights in fragment order |

#### How one MFMA spreads over 64 lanes

You need this picture to follow the rest. For
`mfma_f32_16x16x16bf16_1k`, `D[16×16] = A[16×16] · B[16×16] + C`, every lane
`l` has a column `n = l % 16` and a group `g = l // 16` (0..3):

```
B operand (K × N), 4 bf16 per lane       C / D accumulator (M × N), 4 fp32 per lane

           n = 0    1    2  ...  15                 n = 0    1    2  ...  15
 k  0.. 3:   L0   L1   L2  ...  L15       rows  0.. 3:   L0   L1   L2  ...  L15
 k  4.. 7:  L16  L17  L18  ...  L31       rows  4.. 7:  L16  L17  L18  ...  L31
 k  8..11:  L32  L33  L34  ...  L47       rows  8..11:  L32  L33  L34  ...  L47
 k 12..15:  L48  L49  L50  ...  L63       rows 12..15:  L48  L49  L50  ...  L63

 lane l holds column l % 16,               lane l holds column l % 16,
 4 values starting at k = 4·(l // 16)      4 values starting at row 4·(l // 16)
```

**The two pictures are the same.** The accumulator of one MFMA has exactly
the layout of the **B** operand of the next one, so it can be passed on
without moving any data. (The A operand has rows per lane instead of columns,
so the accumulator does *not* match A.)

The `16x16x32` variant used for QKᵀ looks the same but holds 8 values per
lane, starting at `k = 8·(l // 16)`.

With the MMA instructions FlashAttention-2 was written for it is the other way
round: the accumulator has the layout of the **A** operand, so FA2 computes
`S = Q·Kᵀ` with Q on M and passes P as A to `P·V`. If you copy that on
MI355X, P lands in the wrong lanes, and you need an extra trip through LDS for
every tile.

### 2.1 QSA sparse decode: keep the softmax in registers

#### What the kernel computes

For one row, one KV head and one 16-token tile (dims from section 1.1):

```
S = Q · Kᵀ        Q: 12 heads × 256        K: 16 tokens × 256    S: 12 × 16
P = softmax(S)    (online: running max and sum per head)
O += P · V        V: 16 tokens × 256                             O: 12 × 256
```

Decode has only 1 query token, so the "query" side of the MMA is the 12 heads
of the GQA group. That fits the 16-wide MFMA with 4 padding columns.

#### Before: the original layout

The PR replaces a Triton KV gather plus a decode kernel that was built
around large MMA tiles with the accumulator in TMEM. Moved
to MFMA with FA2's layout (heads on M, P as the A operand), the P tile would
have to be written to LDS and read back in another layout after every
16-token tile.

#### Change: put tokens on M and heads on N, so P is already a B operand

The FlyDSL kernel computes everything transposed: `Sᵀ = K · Qᵀ` and
`Oᵀ += Vᵀ · Pᵀ`.

```
STEP 1   Sᵀ = K · Qᵀ        mfma_f32_16x16x32_bf16, 8 steps over head dim 256

         A = K tile         16 tokens × 32 dims    straight from the KV pool;
                                                   lane: 1 token, 8 contiguous bf16
         B = Qᵀ             32 dims × 16 heads     from LDS (12 heads + 4 padding)
         C = Sᵀ             16 tokens × 16 heads   lane: 1 head, 4 tokens
                                    │
STEP 2   online softmax on C, in registers ──► Pᵀ (convert to bf16)
                                    │
                                    │  same lanes, same layout, no data movement
                                    ▼
STEP 3   Oᵀ += Vᵀ · Pᵀ      mfma_f32_16x16x16bf16_1k, 16 times (16 dims each)

         A = Vᵀ             16 dims × 16 tokens    from LDS, transposed read
                                                   (ds_read_b64_tr_b16)
         B = Pᵀ             16 tokens × 16 heads   ◄── = C of step 1
         C = Oᵀ             16 dims × 16 heads     × 16 tiles = 256 dims
```

Why each choice was made:

- **Tokens on M, heads on N.** One K tile (16 tokens) is used by all 12 heads
  of the group in a single MFMA. Each lane reads one token's 8 contiguous
  bf16 values straight from the pool.
- **Step 1 accumulator = step 3 B operand.** In step 1 each lane ends up with
  1 head × 4 tokens. Step 3 needs B = Pᵀ (tokens × heads), and the B layout
  puts 1 head (column) × 4 tokens (k) in each lane. It is the same data in the
  same lane.
- **`16x16x16` for PV, not `16x16x32`.** The PV MFMA has the tokens on k. The
  `16x16x16` variant takes 4 values of k per lane, which matches the 4 tokens
  in each lane of the step 1 accumulator. The `x32` variant takes 8 per lane
  and would not match.
- **V read transposed.** Step 3 needs Vᵀ as the A operand: each lane needs 4
  tokens of *one* dim, but V is stored by token. `ds_read_b64_tr_b16` reads a
  64-bit piece from LDS and transposes it on the way, the gfx950 version of
  a transposing shared-memory load.

The code, with the three steps marked:

```python
# aiter/ops/flydsl/kernels/qsa/sparse_decode_attn.py
# STEP 1  QK^T: A = K row s straight from the pool, B = Q from LDS
acc = fx.Vector.filled(4, 0.0, fx.Float32)
for step in range_constexpr(HEAD_DIM // 32):
    q_op = _lds_load_vec(
        q_off + (s * HEAD_DIM + 32 * step + 8 * g) * 2, fx.BFloat16, 8
    )
    acc = fx.Vector(
        rocdl.mfma_f32_16x16x32_bf16(
            T.f32x4, [_kv_to_bf16(k_cur[step]), q_op, acc, 0, 0, 0]
        )
    )
# STEP 2  online softmax on acc, in registers
...
p_op = (
    fx.Vector.from_elements(probs, dtype=fx.Float32)
    .to(fx.BFloat16)
    .bitcast(fx.Int16)
)
# STEP 3  PV: the QK accumulator is already the B operand
for m in range_constexpr(16):
    v_op = rocdl.ds_read_tr16_b64(
        fx.Vector.make_type(4, fx.BFloat16),
        fx.to_llvm_ptr(_lds_ptr(v_read_off + m * 512, fx.BFloat16)),
    ).result
    v_op = fx.Vector(v_op).bitcast(fx.Int16)
    new_o.append(
        fx.Vector(
            rocdl.mfma_f32_16x16x16bf16_1k(
                T.f32x4, [v_op, p_op, o_acc[m] * alpha, 0, 0, 0]
            )
        )
    )
```

(`s` = `lane % 16`, `g` = `lane // 16`; `alpha` is the online-softmax rescale
factor; `m * 512` steps over 16 × 16 bf16 V sub-tiles.)

#### How it helps

- P never leaves the registers. There is no LDS write, barrier or read
  between the softmax and PV.
- The output accumulator is 16 tiles × 4 fp32 = **64 registers per lane**.
  That is small enough to leave room for the K/V prefetch buffers of
  section 4. Q is kept in LDS (not registers) for the same reason.
- The cost: 4 of the 16 columns are padding (12 heads), so 25% of each MFMA
  is wasted. That is cheap, because decode is limited by memory, not by
  matrix throughput.

FlashAttention does the same thing in the original kernels. The idea carries
over, but the layout that makes it work is different, and you have to work it out again
for MFMA.

### 2.2 GDN prefill: replace a serial loop with matrix products

#### What the kernel computes

For every 64-token chunk, GDN needs `(I + A)⁻¹`, where `A` (64 × 64) is
strictly lower triangular (zeros on and above the diagonal). It is split into
4 × 4 blocks of 16 × 16:

```
            64 × 64 matrix (I + A), 16 × 16 blocks

          col 0    col 1    col 2    col 3
row 0   [ D0     |  0     |  0     |  0     ]     D = diagonal block:
row 1   [ L10    | D1     |  0     |  0     ]         I + (strictly lower)
row 2   [ L20    | L21    | D2     |  0     ]     L = full block below
row 3   [ L30    | L31    | L32    | D3     ]

  step 1: invert the 4 diagonal blocks D0..D3   ◄── this part is serial
  step 2: fill the blocks below from them with matrix products
```

#### Before: forward substitution

The usual way to invert a triangular block (e.g. `solve_tril` in the FLA
Triton kernels) is forward substitution: row `i` of the inverse needs all rows
before it. For a 16 × 16 block that is **16 dependent steps** of small vector
work. On MI355X this is bad: the matrix cores do nothing during the loop,
and every step waits for the previous one.

#### Change: invert each diagonal block with 6 matrix products

Because `A` is strictly lower 16 × 16, `A¹⁶ = 0`. So the series
`(I + A)⁻¹ = I − A + A² − A³ + … − A¹⁵` ends, and it factors as

```
(I + A)⁻¹ = (I − A) · (I + A²) · (I + A⁴) · (I + A⁸)

  A² = A·A,   A⁴ = A²·A²,   A⁸ = A⁴·A⁴        3 squarings
  then 3 products to multiply the 4 factors    3 products
                                               ─────────────
                                               6 matrix products, all MFMA
```

Multiplying out gives every power of `A` from 0 to 15 exactly once, with the
right sign. That is 16 serial steps replaced by 6 matrix products, and only
the squarings depend on each other (`log₂ 16 = 4` levels).

```python
# aiter/ops/flydsl/kernels/gdr_prefill/gdn_fused_prefill.py
def _diag_inverse(aa, ab, ident):
    """(I + A)^-1 of a strictly lower 16x16 block via (I-A)(I+A^2)(I+A^4)(I+A^8).
    ...
    Products are split-bf16, close to fp32.
    """
```

**Accuracy: split-bf16.** The MFMAs take bf16 (8-bit mantissa), but an
inverse needs about fp32 accuracy. Each fp32 value `x` is split into
`hi = bf16(x)` and `lo = bf16(x − hi)`, and each product becomes 3 bf16
MFMAs:

```
x · y ≈ hi_x·hi_y + hi_x·lo_y + lo_x·hi_y        (lo_x·lo_y is too small to matter)
        └───────── 3 MFMAs (_mm16_x3) ─────────┘
```

This gives close to fp32 accuracy at bf16 MFMA speed.

#### How it helps

The inverse runs on the matrix cores, with no long chain of dependent steps.
This matters because the chunk step is on the critical path: the fused kernel
keeps the state in registers and walks the chunks one after another, so any
time saved per chunk is saved 256 times at 16K tokens.

### 2.3 HC GEMV: store the weights the way the lanes read them

#### What the kernel computes

At decode, the HC site multiplies a few rows of activations (M ≤ 128,
K = 10,240) by a fixed weight matrix. The time goes into reading the weights,
so how each lane reads them decides the speed.

For `mfma_f32_16x16x32_bf16`, one B fragment is a 16 (n) × 32 (k) piece of the
weights. Lane `l` needs column `n = l % 16` and 8 values starting at
`k = 8·(l // 16)`.

#### Before: weights row-major (one row per output)

The original kernel was built for TMA, which copies a whole tile into shared
memory and handles the layout, so the layout in HBM hardly mattered. On MI355X
there is no TMA. Each lane
loads its own piece straight into registers, so the HBM layout decides the
access pattern:

```
row-major w[n][k] (k = 10,240 → row stride 20 KB)

  lanes 0,16,32,48  ──► row n0      bytes 0..63
  lanes 1,17,33,49  ──► row n0+1    bytes 0..63    (20 KB further)
  ...
  lanes 15,31,47,63 ──► row n0+15                 (300 KB further)

  one wave load = 16 pieces of 64 bytes, scattered over 300 KB
```

#### Change: pack each fragment into 1 KB, in lane order

At load time the weights are rearranged once, so each 16 × 32 fragment is one
contiguous 1 KB block, in the order the lanes need it:

```python
# python/sglang/kernels/ops/gemm/hc_fused_flydsl.py
def _pack(w):
    n, k = w.shape
    return w.reshape(n // 16, 16, k // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous()
```

How to read the reshape and permute:

```
w[n, k]  ──reshape──►  [n_tile, n_in(16), k_tile, g(4), e(8)]    k = 32·k_tile + 8·g + e
         ──permute──►  [n_tile, k_tile, g(4), n_in(16), e(8)]
                                       └──── 1 KB fragment ────┘
                                        lane = 16·g + n_in, 8 bf16 each

one fragment in memory:
  [ lane 0: 8 bf16 | lane 1: 8 bf16 | ... | lane 63: 8 bf16 ]   = 64 × 16 B = 1 KB
```

Now lane `l` reads 16 bytes at `fragment_base + 16·l`: one 128-bit buffer
load per lane, and the whole wave reads one contiguous 1 KB block. The next
fragment (next `k_tile`) is the next 1 KB.

#### How it helps

- Every wave load is one contiguous block with full cache lines, not 16
  scattered pieces.
- The address is just `base + 16·lane + 1024·tile`, with no per-lane index
  math in the loop.
- It costs nothing at run time: the packing is done once when the weights are
  loaded, together with the RMSNorm folding from section 2.

### 2.4 Summary of section 3

```
                  original design                    MI355X rewrite
QSA  QK → PV      S = Q·Kᵀ, P passed as A            Sᵀ = K·Qᵀ, Pᵀ passed as B
                  (C layout = A layout on mma)       (C layout = B layout on MFMA)
GDN  (I + A)⁻¹    forward substitution, serial       (I−A)(I+A²)(I+A⁴)(I+A⁸),
                                                     6 split-bf16 matrix products
HC   weights      row-major, TMA handles layout      packed in lane order,
                                                     1 KB per fragment
```

The rule in all three: start from the MFMA lane layout and choose the data
placement to fit it. Do not start from the original kernel's tiles.

## 3. Replace TMA and `cp.async` pipelines with register prefetch

### 3.0 The basic idea: start the load early

A load from HBM takes a long time: many hundreds of cycles before the data
arrives. If a kernel loads a tile and then waits for it, the matrix cores do
nothing during that time:

```
no prefetch      [load 0 ·····wait·····][compute 0][load 1 ·····wait·····][compute 1]
                                         ▲ busy                            ▲ busy

prefetch         [load 0 ·····][load 1 ·····][load 2 ·····][load 3 ·····]
(loads issued                 [compute 0   ][compute 1   ][compute 2   ]
 ahead of time)                ▲ the wait for tile i+1 is hidden behind compute i
```

The original kernels and ours both do this. They differ in **where the tile waits** between the
load and its use, and **how the threads know it has arrived**.

**In the original design**, kernels use a multi-stage pipeline in shared
memory. TMA
or `cp.async` copies tiles from global memory straight into shared memory,
without going through registers. An mbarrier per stage says when a stage is
full (data arrived) or free (consumers are done):

```
           producer (TMA)                         consumers (MMA threads)
                │                                         ▲
                ▼                                         │
  shared mem: [ stage 0 ][ stage 1 ][ stage 2 ][ stage 3 ]   ring of stages
              full/empty mbarrier per stage: the hardware signals "arrived"
```

**On MI355X**, our kernels use the **register file as the staging buffer**.
Every thread issues buffer loads for a tile that will be used *later*. The
values travel through the loop in registers. When their turn comes they are
written to LDS, the workgroup synchronizes, and the MFMAs read the tile:

```
iteration i:   HBM ──buffer load──► registers          (tile i+d, arrives later)
               registers (tile i, loaded d iterations ago) ──► LDS
               ─── barrier ───
               LDS ──► MFMA
```

| | Original design | MI355X (our kernels) |
|---|---|---|
| Copy engine | TMA / `cp.async`, global → shared directly | each thread's own buffer loads, global → registers |
| Where the tile waits | shared memory stages | registers, carried through the loop |
| "Data arrived" | mbarrier, set by hardware | `s_waitcnt` that the compiler puts before first use |
| "Everyone can read it" | mbarrier phase | an ordinary workgroup barrier, **placed by you** |
| Cost | shared memory per stage | registers per prefetched tile |

The last two rows are what make this harder. Each prefetched tile uses
registers, so you have to plan how many tiles to keep in flight. The
barriers are not tied to the data, so you have to work out by hand where they
are needed and where they are not.

Sections 4.1 and 4.2 show the two loops. Section 4.3 shows a buffer-load
feature that removes masks.

### 3.1 GDN prefill: prefetch two chunks ahead, 3 barriers per chunk

#### What the loop does

Each workgroup owns one slice of the GDN state for one sequence and head, and
walks that sequence chunk by chunk, since each chunk's state update needs the
previous one. It has 8
waves in two groups of 4 (section 5). For every 64-token chunk it loads q, k,
v and the gates from HBM, then does a lot of MFMA work in 4 phases. The two
wave groups hand tiles to each other through LDS.

#### Change: the prefetch rotates two register slots

The loop carries the state slice and **two chunks of prefetched inputs** in
its loop-carried values:

```
loop-carried values  =  [ state slice | inputs of chunk ch | inputs of chunk ch+1 ]
                                         (already arrived)     (still in flight)
```

In each iteration the slots rotate:

```
iteration ch
  phase 0   store "inputs of ch" registers → LDS          ← these registers are now free
  phase 1   issue loads for chunk ch+2 into them         ← nxt = pre_next + _prefetch(ch + 2)
  ...
  yield     [ new state | inputs of ch+1 | inputs of ch+2 ]

iteration ch+1
  phase 0   store "inputs of ch+1" → LDS                  (loaded one iteration ago)
  ...
```

The code:

```python
# aiter/ops/flydsl/kernels/gdr_prefill/gdn_fused_prefill.py
def _g_ld(it, off, n, dt):
    return _ld(it, off, n, dt, fx.rocdl.BufferCopy(n * dt.width))

def _prefetch(chunk):
    base = chunk * BT
    tq = _tok(base + r_row)  # clamped to the batch's last token
    qs = [_g_ld(q_it, tq * q_st + 8 * sub + 64 * m, 8, fx.BFloat16) for m in range(2)]
    ks = [_g_ld(k_it, tq * k_st + 8 * sub + 64 * m, 8, fx.BFloat16) for m in range(2)]
    vs = [_g_ld(v_it, tq * v_st + V_EPT * sub, V_EPT, fx.BFloat16)]
    ...
    return qs + ks + vs + [gv, bv]

# Inputs are prefetched two chunks ahead to cover the load latency.
pre0 = _prefetch(fx.Int32(0)) + _prefetch(fx.Int32(1))

for i_t, state in range(fx.Int64(0), fx.Int64(NT), fx.Int64(1), init=s_init + pre0):
    ch = fx.Int32(i_t)
    kbuf = BT * ROW + (ch % 2) * (BT * ROW)  # k tile is double-buffered
    ...
    pre_next = [state[N_S + N_PRE + i] for i in range(N_PRE)]

    # ---- Phase 0: gates, norms, per-token factors, raw q/k to LDS ----
    for m in range_constexpr(2):
        _lds_st(p_qk, r_row * ROW + 8 * sub + 64 * m, qr[m], 8, fx.BFloat16)
        _lds_st(p_qk, kbuf + r_row * ROW + 8 * sub + 64 * m, kr[m], 8, fx.BFloat16)
    ...
    gpu.barrier()                                        # (1)

    # ---- Phase 1: group 0 diag A + inverse and RHS; group 1 q S^T, A tiles ----
    nxt = pre_next + _prefetch(ch + 2)
    ...
    gpu.barrier()                                        # (2)

    # ---- Phase 2: group 0 Tinv row block + vnew; group 1 P^T tiles ----
    ...
    gpu.barrier()                                        # (3)

    # ---- Phase 3: group 0 state update; group 1 output ----
    ...  # reads k[kbuf] transposed; no barrier after this phase

    results = yield new_s + nxt
```

Why it is built this way:

- **Issue the loads in phase 1, not earlier.** Only after phase 0 has stored
  chunk `ch`'s q/k into LDS are those registers free. Reusing them for chunk
  `ch + 2` means the register count does not grow.
- **Two chunks ahead, not one.** Loads issued in phase 1 of chunk `ch` are
  needed in phase 0 of chunk `ch + 2`. In between are phases 1–3 of `ch` *and*
  all of `ch + 1`, almost two chunk steps of MFMA work to hide the HBM latency.
  With only one chunk ahead, there would be just phases 1–3 of one chunk.
- **The wait is automatic.** The compiler puts the `s_waitcnt` for a load just
  before its first use, which is two iterations later. By then the data has
  usually arrived, so the wait costs nothing.

#### Where the barriers go

The two wave groups work on different tiles in the same phase, and hand
results to each other through LDS:

```
             group 0 (waves 0-3)                group 1 (waves 4-7)
phase 0      all 8 waves: gates, norms, q/k registers → LDS
═════════════════════════ barrier (1) ═════════════════════════
phase 1      diag A, inverse, RHS               q·Sᵀ, A tiles
             (+ every wave issues the loads for chunk ch+2)
═════════════════════════ barrier (2) ═════════════════════════
phase 2      Tinv row block, vnew      ◄──►     Pᵀ tiles
═════════════════════════ barrier (3) ═════════════════════════
phase 3      state update (reads k)             output
             (no barrier here)
                     │
                     ▼  next iteration, phase 0
```

- **Barrier (1): after the registers → LDS stores, before the first MFMA.**
  Each thread stored only its own slice of q/k, but every wave reads the whole
  tile.
- **Barriers (2) and (3): between phases that hand tiles across groups.**
  Group 0 writes the inverse and `vnew` that group 1 reads, and the other way
  round.
- **No barrier at the end of the iteration.** That would be a fourth barrier.
  It is not needed because of the k double buffer (next part).

#### Why k is double-buffered, and only k

Without a barrier at the end, a fast wave can start the next iteration while a
slow wave is still in phase 3:

```
single k buffer:

  slow wave   ... phase 3 of ch: reading k(ch) from LDS[k] ...
  fast wave   ... phase 0 of ch+1: writing k(ch+1) to LDS[k]     ✗ overwrites data
                                                                   still being read
two k buffers, picked by ch % 2:

  slow wave   ... phase 3 of ch:   reads  LDS[k, buffer 0]
  fast wave   ... phase 0 of ch+1: writes LDS[k, buffer 1]        ✓ no conflict
```

Two buffers are always enough. The fast wave cannot get past barrier (1) of
`ch + 1` until the slow wave arrives there too, so the waves are never more
than one iteration apart.

Only k needs this, because only k is read in phase 3. Every other tile that
phase 0 writes (q, v, the per-token factors) is last read before barrier (3).
When a fast wave writes them again in the next phase 0, every wave has
already passed barrier (3), so nobody still reads them.

Result: **3 barriers per chunk instead of 4**, for the cost of one extra
64 × 136 bf16 k tile in LDS.

### 3.2 QSA sparse decode: prefetch one tile ahead, no barrier in the loop

#### What the loop does

Each workgroup of 4 waves handles one split of KV positions (section 1.1).
The waves take turns over the 16-token tiles:

```
split = 192 positions = 12 tiles of 16

wave 0: tiles 0, 4, 8          wave i, iteration it → tile (i + it · 4)
wave 1: tiles 1, 5, 9
wave 2: tiles 2, 6, 10
wave 3: tiles 3, 7, 11
```

#### Change: prefetch the next tile into registers

While one tile is being computed, the K/V loads for the wave's next tile are
already in flight:

```python
# aiter/ops/flydsl/kernels/qsa/sparse_decode_attn.py
# Software pipeline: the loads for tile i+1 are in flight while tile i
# computes. The last trip re-reads its own (cache-hot) tile instead of
# branching around the prefetch.
next_it = (it_i32 + 1 < n_iters).select(it_i32 + 1, it_i32)
prefetched = _load_tile((warp + next_it * NUM_WAVES) * TILE_TOKENS)
```

```
wave 0:   it = 0              it = 1              it = 2 (last)
compute   tile 0              tile 4              tile 8
prefetch  tile 4              tile 8              tile 8 again (cache hit, discarded)
```

- **Last trip: reload, don't branch.** On the last iteration there is no next
  tile. Instead of an `if` around the prefetch, the wave loads its own tile
  again. It is already in cache, so it is cheap, and the loop body stays the
  same in every iteration.
- **Q stays in LDS.** The prefetched K/V tile needs registers. Q is the same
  for every tile, so it is kept in LDS and read from there (section 3.1), which
  leaves the registers for the prefetch.

#### Why the main loop needs no barrier

A barrier is only needed when one wave reads LDS that **another** wave
wrote. In QSA each wave stages its V tile in **its own** LDS region and is the
only one that reads it back (for the transposed read of section 3.1). Inside
one wave, LDS accesses complete in order, so no barrier is needed:

```
LDS layout of one workgroup

  [ slots (all waves) ][ Q (all waves) ][ V wave 0 ][ V wave 1 ][ V wave 2 ][ V wave 3 ]
   └─── written once in the prologue ──┘ └───────── private to each wave ──────────┘
        → barrier after the prologue          → no barrier in the main loop

epilogue: the 4 waves merge their partial results through the V region
        → barriers again (the region is shared now)
```

So QSA has barriers only in the prologue (slots and Q are written by all
threads and read by all) and in the epilogue (the waves merge their results
and reuse the V region).

Compare with GDN in 4.1: there the waves *must* share tiles, because the two
groups split the work of one chunk, so barriers are needed. In QSA the waves
work on different tokens and share nothing until the end.

### 3.3 Use the out-of-bounds behavior of buffer loads and stores

#### What a buffer access is

On AMD, a buffer load or store goes through a **buffer descriptor**: a base
address plus `num_records`, the size of the buffer. The hardware checks every
access against that size:

```
descriptor: base ─────────────────► base + num_records
            [  valid range                            ]  past the end:
                                                         load  → returns 0
                                                         store → dropped
```

So the end of a tensor is handled by the hardware, and the kernel needs no
per-lane `if (row < n)` masks.

#### GDN: the last chunk of a sequence

Sequences are packed back to back, and a sequence's length is usually not a
multiple of 64. So the last chunk runs past the end of the sequence:

```
packed tokens:   [ ········ seq 0 ········ | ······ seq 1 ······ ]
seq 0 chunks:    [ chunk 0 ][ chunk 1 ][ chunk 2 ]
                                       [ 40 rows | 24 rows      ]
                                         seq 0     belong to seq 1
```

The output tile of the last chunk has 64 rows, but only 40 belong to seq 0.
If the kernel wrote all 64, it would overwrite seq 1's output, which another
workgroup writes. So the output descriptor is cut at the end of this sequence:

```python
# aiter/ops/flydsl/kernels/gdr_prefill/gdn_fused_prefill.py
# Only o sets num_records: the last chunk's rows past eos
# belong to the next sequence, and the hardware drops those stores.
o_rec = fx.Int64(T_local) * (H * VD * 2)
o_it = _view(o_t, (bos64 * H + fx.Int64(i_h)) * VD + fx.Int64(i_v) * BV, o_rec)
```

```
o descriptor:  base = row bos of seq 0, num_records = T_local rows
               [ rows of seq 0 ]  rows 40..63 of the last chunk → stores dropped
```

#### HC: pad the decode rows to the 16-row tile

An MFMA always works on 16 rows (section 3), but at decode there may be only
1–3 rows. The rows past `M` must read as zero. HC points them at the first
address past the end of the activations, so the hardware returns 0:

```python
# python/sglang/kernels/ops/gemm/hc_fused_flydsl.py
def a_offset(row, kcol):
    # Descriptor-bound OOB loads return zero for the padded M rows.
    return (row < M).select(fx.Int32(row * K + kcol), fx.Int32(M * K))
```

```
activations, M = 3 real rows, MFMA tile = 16 rows

row  0  [ real data ]  ◄── offset row·K + kcol
row  1  [ real data ]
row  2  [ real data ]
row  3  [ 0 0 0 ... ]  ◄── offset M·K = first byte past the end → load returns 0
 ...
row 15  [ 0 0 0 ... ]
```

The padded rows give zero results, which are never stored. There is no
branch, and every lane runs the same code.

### 3.4 Summary of section 4

```
                    original design                  MI355X rewrite
staging buffer      shared-memory stages (TMA)       registers, carried through the loop
"data arrived"      mbarrier per stage               s_waitcnt, placed by the compiler
"safe to read"      mbarrier phase                   workgroup barrier, placed by you
GDN chunk loop      multi-stage TMA pipeline         2 chunks ahead, 3 barriers,
                                                     k double-buffered
QSA tile loop       multi-stage TMA pipeline         1 tile ahead, wave-private LDS,
                                                     no barrier in the loop
tails and padding   TMA OOB fill / masks             buffer descriptor: OOB load = 0,
                                                     OOB store dropped
```

The rule: decide for every LDS tile who writes it and who reads it. Put a
barrier only where a wave reads another wave's data. Double-buffer a tile
when skipping a barrier would let a fast wave overwrite it.

### Monokernel

When we first benchmarked this model on AMD MI355X we quickly realized that the low concurrency region (i.e., <= 8) was going to be the most critical for us in terms of performance.

![Traces before monokernel](/images/blog/qwen38-flash-next-mi355x/before_mono.png)
*Figure 3: Kernel trace of a full decode step at concurrency 1 without monokernel*

Figure 3 shows the full trace of one decode step at concurrency 1. As you can see, it takes roughly 14ms for a total of over **1600 individual kernel launches**.
Launching a lot of kernels is not necessarily a problem as HIP graphs can get rid of kernel launch overheads but in this case it actually is. The issues is that these kernels are all very small so the GPU is not doing useful work but mostly loading data from and to HBM memory. In this scenario, the most practical optimization strategy is simply trying to reduce as much as possible the HBM traffic, this means fusing multiple operations together.

In order to achieve the best possible performance at low concurrency, we decided to fuse as many kernels as possible into a single (mono) kernel.
As we saw in figure 2, the basic building block for Qwen3.8-Flash-Next is 3x GDN layers followed by a QSA layer, repeated 12 times for a total of 48 hidden layers. We implemented a specialized FlyDSL kernel that replaces this basic building block with a **single kernel launch**.

![Traces with monokernel](/images/blog/qwen38-flash-next-mi355x/after_mono.png)
*Figure 4: Kernel trace of a full decode step at concurrency 1 with the new FlyDSL monokernel*

Figure 4 shows a full decode step with the monokernel on. With this we got rid of over 1600 kernel launches and replaced them with just **13** and went from almost **14ms to 4ms**, that's over **3x improvement** in decode speed.

For more details, the monokernel source code is available on [GitHub](https://github.com/sgl-project/sgl-amd-mono/tree/main/sgl_amd_mono/qwen38_flash_next).

## Conclusion


## Acknowledgements

We thank the Qwen team, the SGLang community, and the AMD AITER, FlyDSL, Quark, ROCm and InferenceX contributors whose work this post builds on. [TODO: names.]

## Test configuration and disclaimer

**Software stack of the recipes above:** `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261117` (SGLang `c67db7c762`, AITER `581579ab`, FlyDSL 0.4.2), patched with the open PRs listed above. **Numbers in Performance and Accuracy** were measured on one GPU of an 8x AMD Instinct MI355X node in October 2026 with `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261026` (SGLang `5ef99013`, AITER `285579b6`), patched with the PRs open at the time, through the InferenceX AgentX agentic-coding harness (AIPerf replay of recorded coding-agent traces, 3,600 s per point, InferenceX run [40420614823](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40420614823)). Per-PR numbers quoted in the optimization sections come from each pull request's own benchmarks, on the base and image given there. Speculative-decoding throughput uses InferenceX's committed synthetic acceptance length; accuracy uses lm-eval with real verification. Results on other hardware, software versions, settings or workloads will differ; these figures are for illustration and are not a performance guarantee.

#### Article by

Authors: Nikolai (@amd-nprotaso), Christian Gilli (@amd-cgilli), and Chang Liu (@ChangLiu0709) — AMD.
