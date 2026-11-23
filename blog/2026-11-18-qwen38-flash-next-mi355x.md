---
title: "Qwen3.8-Flash-Next on AMD Instinct MI355X: serving and optimization with SGLang"
author: "Nikolai (@amd-nprotaso), Christian Gilli (@amd-cgilli), and Chang Liu (@ChangLiu0709) — AMD"
date: "November 18, 2026"
previewImg: /images/blog/qwen38-flash-next-mi355x/preview.png
type: blog
---

Qwen3.8-Flash-Next is the first open-weight release of the efficient architecture that will underpin Qwen4: a 125B-parameter Mixture-of-Experts model that activates only 6B parameters per token and is built for long-context agentic workloads. [One sentence for its community impact, and the reason binding to its model architecture, features, dataset(if involved), etc]. Thus, enablement and performance optimization of Qwen3.8-Flash-Next on MI355X is important. In our blog, we show how to optimize the inference of Qwen3.8-Flash-Next by ... We benchmark the inference performance on InferenceX AgentX, which [one sentence to introduce the dataset and benchmark metrics], outperforming our competitor NVIDIA B200 and B300 across all concurrency levels.

This post describes how we serve Qwen3.8-Flash-Next on AMD Instinct MI355X GPUs with SGLang. We start with our latest performance result on InferenceX AgentX, then reveal the optimization behind the scene by briefly introducing the Qwen model architecure, the key optimizations, and the ideas of using AI agent to automate the process. All the results can be reproduced by referring to the command on SGLang cookbook [link].

## Performance

We used InferenceX AgentX as our benchmark dataset and metrics, as [reasons]. As shown in Figure 1, [the overall configurations of the run, including concurrency list as ...] and [AgentX configurations, like including how many requests, benchmark time duration as 3600s]. [Analysis of the overall trend]. 

// screenshot from InferenceX

*Figure 1 [TODO: chart]: throughput per GPU against interactivity across 1–16 agents.*

[Analysis of the high concurrency scenario, with the optimizations, i.e., kernel fusion, xxx, (summary from our optimizations including the major software stacks)]. [Analysis of the low concurrency scenario: with monokernel, xxx]. We are going to give more details about the optimizations in the latter sections.

**NOTE:** Until the day Nov 19, 2026, we wrote the blog, the performance submission shown in the figure depends on 7 PRs [PR links] that awaits to be merged to the SGLang main branch.

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

## 1. Size the grid for 256 CUs, not for B300's SMs

**On NVIDIA** the launch shape of a kernel, meaning the split count, the tile
sizes and the number of programs per row, is tuned for B300, which has up to
160 SMs. Often these values are hard-coded in the launch heuristics.

**On MI355X** there are 256 CUs, at least 1.6× as many. A grid that fills a
B300 fills only about 60% of an MI355X. At decode the grid is often even
smaller. The old QSA decode ran one program per (row, KV head). Qwen3.8 has 2
KV heads at TP1, so 8 decode rows gave 16 programs and left 240 CUs idle.

**Rewrite:** split the long dimension until the grid fills the GPU, then add
a small reduce step. Compute the split count from the CU count at run time.
This is how the QSA kernel picks its KV split:

```python
# aiter/ops/flydsl/qsa_sparse_decode.py
# Measured on MI355X (Hkv in {1, 2}, batch 1-128): one CTA per CU beats two, since
# every extra split adds a prologue and a partial to merge; and below two 16-token
# tiles per wave the fixed per-CTA latency outweighs the extra parallelism.
_CTAS_PER_CU = 1
_MIN_TOKENS_PER_SPLIT = 2 * SPLIT_TOKEN_ALIGN

def recommended_num_splits(
    num_rows: int, num_kv_heads: int, topk: int, device: torch.device
) -> int:
    """KV splits that fill the GPU at this batch; fixed per CUDA-graph batch size."""
    _, num_cus = _device_info(torch.device(device).index or 0)
    wanted = -(-(_CTAS_PER_CU * num_cus) // max(1, num_rows * num_kv_heads))
    wanted = min(wanted, max(1, -(-topk // _MIN_TOKENS_PER_SPLIT)))
    return _split_layout(topk, wanted)[1]

# the kernel launch
sparse_decode_attn_kernel(...).launch(
    grid=(num_rows, num_kv_heads, num_splits),
    block=(BLOCK_THREADS, 1, 1),
    stream=stream,
)
```

For 8 rows and top-k 2051 this gives 11 splits, which is 176 CTAs instead of
16. A small reduce kernel merges the partial results. It is skipped when there
is only one split, which happens at 128 rows.

GDN prefill has the opposite problem. Each workgroup walks its sequence's
chunks serially, so a long sequence with few heads makes a small grid. The
wrapper picks the V slice width from the CU count. A gate sends grids that are
still too small to the multi-kernel path:

```python
# aiter/ops/flydsl/gdn_fused_prefill.py
def _pick_bv(N: int, H: int, device: torch.device) -> int:
    """V slice per workgroup.

    LDS allows one workgroup per CU, and each walks its sequence's chunks
    serially. When the BV=16 grid still fits in one pass over the CUs, the
    narrower slice shortens every chunk step; otherwise BV=32 recomputes the
    V-independent work half as often.
    """
    return 16 if (_VD // 16) * N * H <= _num_cus(device) else 32

# in gdn_fused_prefill_supported(...):
    # The fused kernel walks each sequence's chunks serially at ~2.5 us per
    # chunk whatever H is; the multi-kernel path spreads all chunks over the
    # GPU and serializes only its state scan. Thresholds measured on MI355X ...
    if max_seqlen is not None and not (max_seqlen <= 1024 or H * T >= 24 * max_seqlen):
        return False
```

For Qwen3.8 at TP1 (48 value heads) the gate never declines a batch. At TP8
there are only 6 value heads, and one 16K sequence is 48 workgroups on 256 CUs.
There the fused kernel is 0.63× and the gate sends the batch to the
multi-kernel path.

## 2. Remove the extra passes around the main kernel

**Typical structure:** the main kernel is surrounded by helper passes. In
QSA, a Triton kernel gathered each row's 2,051 selected KV positions into a
packed buffer so that a standard dense decode kernel could run on it. In GDN,
the staged path wrote every per-chunk state to HBM and read it back in a
separate output kernel. Each pass adds a round trip through HBM and a launch.

**On MI355X**, at decode sizes, these passes became the main cost. QSA spent
39 µs of its 120 µs on the gather alone. Staged GDN took 8 launches per layer.

**Rewrite:** fuse into one kernel and keep the intermediates on chip. The QSA
kernel does the lookup from index to pool slot itself. Each CTA resolves its
share of the 2,051 positions once and keeps the slots in LDS, so the gather
pass and its scratch buffer go away:

```python
# aiter/ops/flydsl/kernels/qsa/sparse_decode_attn.py
# ---- Prologue: logical positions -> pool slots, staged in LDS ----
for j in range_constexpr(prologue_iters):
    local = tid + j * BLOCK_THREADS
    tok = split_start + local
    in_range = (local < tokens_per_split) & (tok < topk)
    safe_tok = fx.Int64(in_range.select(tok, zero_i32))
    pos = fx.ptr_load(fx.add_offset(indices, idx_row + safe_tok))
    valid = in_range & (pos >= 0) & (pos < seq_len)
    safe_pos = fx.Int64(valid.select(pos, zero_i32))
    slot = fx.ptr_load(fx.add_offset(req_to_token, r2t_row + safe_pos))
    slot = valid.select(slot, fx.Int32(-1))
    ...
    _lds_store(slot_off + local * 4, fx.Int32, slot)
```

In GDN the state slice stays in MFMA accumulators across all chunks. The work
that doesn't depend on V (l2norm, `k·kᵀ`, the triangular inverse) is
recomputed by every V slice, which is cheaper than a round trip through HBM.
At 16K tokens one layer becomes 1 launch and 767 µs instead of 8 launches and
1,227 µs.

At the HC site the fusion is pure algebra, and it would work on any vendor.
The RMSNorm weight and the inject projection are folded into the packed down
weights once, at load time:

```python
# python/sglang/kernels/ops/gemm/hc_fused_flydsl.py
class HcFusedWeights:
    def __init__(self, down, up, inject, norm_weight, eps):
        ...
        self.w1 = (1.0 + norm_weight.float()).contiguous()
        rows = [down.float() * self.w1]
        if inject is not None:
            self.ni = inject.shape[0]
            rows.append(inject.float() * self.w1)
        d = torch.cat(rows)
```

The per-branch `rsqrt` is applied to each split-K partial in the reduce
kernel. A split never covers two branches, so this works:

```python
# _hc_reduce_fused
per = SPLIT // _HC
acc = fx.Float32(0.0)
for b in range_constexpr(_HC):
    total = fx.Float32(0.0)
    for j in range_constexpr(NP):
        total = total + _load(ss, row * (_HC * NP) + (b * NP + j), 1, fx.Float32)[0]
    scale = fx.Float32(1.0) / math.sqrt(total * fx.Float32(_HC / K) + fx.Float32(EPS))
    part = fx.Float32(0.0)
    for s in range_constexpr(b * per, (b + 1) * per):
        part = part + _load(p, fx.Int32(s * M * R) + i, 1, fx.Float32)[0]
    acc = acc + part * scale
```

The sums of squares `ss` come from the previous site's combine
(`hc_apply_inject`). It computes them while it writes the residual, so the
next norm needs no extra pass over the data. The same idea was proposed for
SM100 in [sglang#40552](https://github.com/sgl-project/sglang/pull/40552).

## 3. Rewrite the MMA part for MFMA, don't translate it

**On NVIDIA** B300 kernels are built around tcgen05 MMAs. A single thread
issues one MMA for the whole CTA on large tiles (128×N, or up to 256×N across
a CTA pair). TMA brings the operands into shared memory, and the accumulator
sits in Tensor Memory (TMEM, 256 KB per SM), outside the register file.

**On MI355X** each wave of 64 lanes issues its own MFMA on 16×16 tiles. Both
the operands and the accumulators are in registers, so a large accumulator
tile uses up the register budget that TMEM keeps free on B300. The lane layout
(`lane % 16`, `lane // 16`) is also different. A tile layout or fragment
mapping copied from a CUDA kernel will be wrong or slow.

**Rewrite:** choose the operand placement so the data flows from one MFMA to
the next without going through LDS. In QSA, the 16 tokens of a tile go on M
and the 12 query heads of a KV group go on N, so one K tile serves the whole
group. The QK accumulator then has exactly the B-operand layout of the PV
MFMA, so the softmax probabilities never leave registers. V is read back
transposed with `ds_read_b64_tr_b16`, the gfx950 counterpart of
`ldmatrix.trans`.

```python
# aiter/ops/flydsl/kernels/qsa/sparse_decode_attn.py
# QK^T: A = K row s straight from the pool, B = Q from LDS
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
...  # online softmax on acc, in registers
p_op = (
    fx.Vector.from_elements(probs, dtype=fx.Float32)
    .to(fx.BFloat16)
    .bitcast(fx.Int16)
)
# PV: the QK accumulator is already the B operand
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

FlashAttention does the same on NVIDIA, but you have to find the matching
layout again for MFMA.

Two more patterns from our kernels:

- **Avoid serial steps.** GDN needs `(I + A)⁻¹` of a strictly lower 64×64
  block for every chunk. Instead of a serial forward substitution, each 16×16
  diagonal block is inverted with matrix products only:

  ```python
  # aiter/ops/flydsl/kernels/gdr_prefill/gdn_fused_prefill.py
  def _diag_inverse(aa, ab, ident):
      """(I + A)^-1 of a strictly lower 16x16 block via (I-A)(I+A^2)(I+A^4)(I+A^8).
      ...
      Products are split-bf16, close to fp32.
      """
  ```

  Each product runs as 3 bf16 MFMAs on hi/lo halves (`_mm16_x3`). This gives
  close to fp32 accuracy at bf16 MFMA speed.
- **Pack weights in fragment order.** The HC weights are packed once so that
  one 16×32 B fragment is 64 lanes × 8 bf16. Each lane then gets its part
  with one 128-bit buffer load:

  ```python
  # python/sglang/kernels/ops/gemm/hc_fused_flydsl.py
  def _pack(w):
      n, k = w.shape
      return w.reshape(n // 16, 16, k // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous()
  ```

## 4. Replace TMA and `cp.async` pipelines with register prefetch

**On NVIDIA** latency is hidden by multi-stage pipelines. TMA or `cp.async`
copies tiles from global memory straight into shared memory, and an mbarrier
per stage says when a stage is full or free.

**On MI355X** our kernels use the register file as the staging buffer. Every
thread issues buffer loads for a tile that will be used *later*. The loaded
values are carried through the loop in registers. When their turn comes they
are written to LDS, the workgroup synchronizes, and the MFMAs read the tile.
The synchronization is now an ordinary workgroup barrier, so you have to place
the barriers yourself.

This is the chunk loop of the fused GDN kernel. Inputs are prefetched two
chunks ahead:

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

Where the barriers go:

- **After the registers → LDS store, before the first MFMA (1).** All 8 waves
  read the whole q/k tile, but each thread wrote only its own slice.
- **Between phases that hand tiles from one wave group to the other (2, 3).**
  Group 0 writes the inverse and `vnew` that group 1 reads, and the other way
  round.
- **Not at the end of the iteration.** Without double buffering we would need
  a fourth barrier there. The next phase 0 would overwrite the k tile while
  phase 3 of the slower waves still reads it. Every other single-buffered tile
  (q, v, the per-token factors) is last read before barrier (3), so only k
  needs two buffers. That makes 3 barriers per chunk instead of 4.
- **Issue the next loads after phase 0.** By then chunk `ch`'s q/k registers
  have been stored to LDS and are free, so chunk `ch + 2` reuses them and the
  register count doesn't grow. The loads overlap with all the MFMA work of
  phases 1–3. The compiler puts the wait for a load (`s_waitcnt`) just before
  its first use, which is two iterations later, so by then the data has
  usually arrived.
- **No barrier when LDS is private to one wave.** In QSA each wave stages its
  V tile in its own LDS region and is the only one that reads it back. Inside
  one wave, LDS accesses complete in order, so the main loop has no barrier at
  all. QSA needs barriers only in the prologue (slots and Q staged by all
  threads) and in the epilogue, where the four waves merge their results and
  reuse the V region.

QSA prefetches the next 16-token K/V tile into registers in the same way.
On the last trip it reloads its own tile (it's already in cache) instead of
branching around the prefetch. Q is kept in LDS to free registers for the
prefetch:

```python
# aiter/ops/flydsl/kernels/qsa/sparse_decode_attn.py
# Software pipeline: the loads for tile i+1 are in flight while tile i
# computes. The last trip re-reads its own (cache-hot) tile instead of
# branching around the prefetch.
next_it = (it_i32 + 1 < n_iters).select(it_i32 + 1, it_i32)
prefetched = _load_tile((warp + next_it * NUM_WAVES) * TILE_TOKENS)
```

**Use the out-of-bounds behavior.** A buffer load past the end returns 0,
and a store past `num_records` is dropped. GDN uses this for the last chunk of
each sequence, with no per-lane masks:

```python
# aiter/ops/flydsl/kernels/gdr_prefill/gdn_fused_prefill.py
# Only o sets num_records: the last chunk's rows past eos
# belong to the next sequence, and the hardware drops those stores.
o_rec = fx.Int64(T_local) * (H * VD * 2)
o_it = _view(o_t, (bos64 * H + fx.Int64(i_h)) * VD + fx.Int64(i_v) * BV, o_rec)
```

HC uses the same rule to pad decode rows up to the 16-row MFMA tile:

```python
# python/sglang/kernels/ops/gemm/hc_fused_flydsl.py
def a_offset(row, kcol):
    # Descriptor-bound OOB loads return zero for the padded M rows.
    return (row < M).select(fx.Int32(row * K + kcol), fx.Int32(M * K))
```

## 5. Rewrite the warp-level code for wave64

- **Warp 32 → wave 64.** Every `__shfl_xor` reduction, lane index map and
  "one warp per row" assumption changes.
- **Use DPP for row reductions and scans.** GDN computes the in-chunk
  cumulative decay over 64 tokens with DPP instead of a chain of shuffles:

  ```python
  # aiter/ops/flydsl/kernels/gdr_prefill/gdn_fused_prefill.py
  def _wave_scan(x):
      """Inclusive wave64 prefix sum (gfx9 DPP row_shr + row_bcast)."""
      s = x
      for i in range(4):
          s = s + _dpp_f32(s, 0x110 + (1 << i))  # row_shr
      s = s + _dpp_f32(s, 0x142, row_mask=0xA)  # row_bcast:15
      return s + _dpp_f32(s, 0x143, row_mask=0xC)  # row_bcast:31
  ```

- **Keep wave-uniform values in scalar registers:**

  ```python
  # Wave index as a scalar so per-wave selects and addresses stay in SGPRs.
  w = fx.Int32(rocdl.readfirstlane(T.i32, (tid // WAVE).ir_value()))
  ```

- **Warp specialization.** gfx950 has no `setmaxnreg` and no async barriers.
  GDN uses two wave groups, with one wave of each group on every SIMD, so both
  groups issue MFMAs at the same time:

  ```python
  # Two wave groups of four (one wave of each per SIMD): group 0 owns the state
  # slice and the triangular inverse, group 1 the q-side tiles and the output.
  NUM_WAVES = 8
  GROUP = 4
  ```

## 6. Plan for 160 KB LDS

gfx950 has 160 KB LDS per CU. B300 has up to 228 KB of shared memory per SM,
plus 256 KB of TMEM for accumulators. Tiles sized for B300 may not fit, or
may fit only one workgroup per CU, as in GDN. That brings you back to rule 1.
Pad the LDS rows against bank conflicts:

```python
# aiter/ops/flydsl/kernels/gdr_prefill/gdn_fused_prefill.py
# Row pitch of the [token][K] and [V][K] bf16 LDS tiles; the 8-element pad
# shifts consecutive rows by 4 banks so a 16-row b128 operand read is
# conflict-free.
ROW = KD + 8
```

## 7. Don't lose the gain on the host

FlyDSL specializes each kernel at compile time (GQA group, top-k, dtypes, row
count). That is good for the kernel, but every new shape is a new compile, and
calling the JIT function re-traces it every time. The HC kernels handle both
problems:

```python
# python/sglang/kernels/ops/gemm/hc_fused_flydsl.py
def _bucket_rows(m):
    """M is a Constexpr, so each distinct row count is a compile. Round up to a power of
    two within the first MFMA tile, then to whole tiles: the kernels already compute
    whole tiles, so padding adds no MFMA work and M = 1 stays exact."""
    if m <= _MFMA_M:
        return 1 << (m - 1).bit_length()
    return -(-m // _MFMA_M) * _MFMA_M

# Calling a JitFunction re-traces it to MLIR on every call, which costs far more than the
# kernels outside CUDA graphs; dispatch a CompiledFunction per Constexpr set instead.
_compiled = {}
```

Also keep everything CUDA-graph safe. The QSA split count is fixed per batch
size, and the workspace is allocated once in SGLang:

```python
# python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py
# Sized once for every captured batch: graphs keep raw pointers to it, so
# it is never reallocated. A larger eager batch gets a per-call workspace.
if self._flydsl_decode_workspace is None:
    capacity = max(
        partials,
        flydsl_decode.workspace_partials(
            self._cuda_graph_max_tokens, num_q_heads, num_kv_heads, topk, device
        ),
    )
```

## What you can keep

The algorithm, online softmax, chunked recurrences and the algebraic fusions
all carry over. Keep the existing path as the fallback. Each of our kernels
checks its operands and sends everything else to the old path:

```python
# python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py
flydsl_decode = _resolve_flydsl_sparse_decode()
if flydsl_decode is not None and flydsl_decode.can_use_flydsl_qsa_sparse_decode(
    q, k_buffer, v_buffer
):
    return self._forward_flydsl_fused_sparse(...)
# everything else: the existing Triton gather + decode
```

Validate against an fp32/fp64 reference rather than bit-exactness. All three
rewrites came out within 1 bf16 ulp of the old path or more accurate than it.
For example, the QSA relative L2 error went from 4–6e-3 to 1.7e-3, and GSM8K
did not change.

## Checklist

1. Does the grid fill 256 CUs at your real decode batch? If not, split the
   KV or K dimension, with the split count taken from the CU count.
2. Are there gather, scatter or staging passes around the main kernel? Fuse
   them in.
3. Is the MMA mapping written for MFMA 16×16 tiles, with accumulators chained
   from one MFMA to the next?
4. Are TMA pipelines replaced with register prefetch and buffer loads? Is
   there a barrier after every shared LDS tile is written, and is the tile
   that the last phase still reads double-buffered?
5. Are the reductions and lane maps written for wave64?
6. Does the LDS use fit in 160 KB, and is it padded against bank conflicts?
7. Are the kernels cached and CUDA-graph safe, with a fallback for the shapes
   where the new kernel loses?



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
