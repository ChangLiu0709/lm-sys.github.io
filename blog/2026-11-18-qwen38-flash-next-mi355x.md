---
title: "Qwen3.8-Flash-Next on AMD Instinct MI355X: serving and optimization with SGLang"
author: "Chang Liu (@ChangLiu0709), Nikolai (@amd-nprotaso), and Christian Gilli (@amd-cgilli) — AMD"
date: "November 18, 2026"
previewImg: /images/blog/qwen38-flash-next-mi355x/preview.png
type: blog
---

*Draft, 2026-10-28. Items marked **[TODO]** need a measurement or a decision before publication.*

## TL;DR

- Qwen3.8-Flash-Next, a hybrid linear/sparse-attention Mixture-of-Experts model with native multi-token prediction (MTP), runs on a **single** AMD Instinct™ MI355X GPU with SGLang. The model's 36 Gated DeltaNet (GDN) layers, 12 QuerySparse Attention (QSA) layers, hyper-connections, 512-expert MoE and the MTP draft all run on AITER and FlyDSL kernels.
- An AMD Quark checkpoint with MXFP4 experts (including the MTP draft), per-channel FP8 attention and an FP8 per-layer-embedding (PLE) table fits in about 116 GiB. Offloading the PLE table to CPU pinned memory cuts GPU weight memory from about 130 GB to 82 GB and grows the KV-cache pool from about 2.1M to 3.7M tokens.
- At 1–8 concurrent users, a FlyDSL **monokernel** runs the whole MTP target-verify step in 13 persistent launches and each draft step in one launch, instead of hundreds of small kernels.
- Fused FlyDSL kernels replace the hot decode-path operators: QSA sparse decode (120 → 33 µs per call, +10% output throughput at 32k/32k with 8 users), the hyper-connection input mix (1.7–2.4x per call), GDN conv1d and decode/verify, and tuned FP8/BF16 GEMM tables.
- On the InferenceX AgentX agentic-coding benchmark (real coding-agent traces, one hour per point), one MI355X serves **16,613 total (input + output) tok/s at 407 tok/s per user** for one agent, and **77,097 tok/s at 122 tok/s per user** for 16 agents. GSM8K is 97.3% (strict match).
- Every change is in upstream SGLang, AITER or sgl-amd-mono pull requests, applied on top of the public `lmsysorg/sglang-rocm` image; no custom image is needed.

## Introduction

Qwen3.8-Flash-Next is [TODO: one-line model positioning from the Qwen team, e.g. "the Qwen team's efficient flagship for coding and agent workloads"]. Its architecture combines three ideas that each need their own kernels: linear attention through Gated DeltaNet for most layers, a learned sparse attention (QSA) for the rest, and hyper-connections that carry four copies of the hidden state between layers. Agentic coding adds a fourth requirement: very long, mostly shared prompts (tens to hundreds of thousands of tokens) with short, latency-sensitive replies.

This post describes how we serve Qwen3.8-Flash-Next on AMD Instinct MI355X GPUs with SGLang: the quantized checkpoint, the kernels we wrote or tuned in AITER and FlyDSL, the FlyDSL monokernel for speculative decoding at low concurrency, the measured performance on the InferenceX AgentX benchmark, and a serving recipe.

## Qwen3.8-Flash-Next at a glance

For inference, five properties matter:

- **Hybrid attention.** 36 of the 48 layers use Gated DeltaNet, a linear-attention layer with a fixed-size recurrent state per request, so its decode cost does not grow with context. The other 12 use QuerySparse Attention: each query attends to a learned selection of at most 2,051 past positions (512 compressed blocks of 4, plus 3), with 24 query heads sharing 2 KV heads of dimension 256.
- **Hyper-connections.** Instead of one residual stream, the model keeps 4 streams of width 2,560. Before every block a gated mix (a 10,240 → 320 → 10,240 low-rank gate) collapses them into one input. It is small but runs in every layer at every step.
- **Sparse MoE.** 512 routed experts with a softmax router. [TODO: active experts per token, shared expert, total and active parameters.]
- **Per-layer embeddings (PLE).** A large n-gram embedding table feeds every layer. It is read sparsely, which makes it a good candidate for host memory.
- **Native MTP.** A built-in draft layer proposes tokens for speculative decoding (SGLang `NEXTN`), verified by the target in one forward pass.

## The checkpoint: MXFP4 experts with AMD Quark

We serve [`amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8`](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8), produced with AMD Quark:

| Component | Format |
|---|---|
| Routed experts (target and MTP draft) | OCP MXFP4 (FP4 with one E8M0 scale per 32 values) |
| Attention projections | FP8, per-channel scales |
| PLE embedding table | FP8 (`float8_e4m3fn`) |
| Router, norms, LM head | High precision |

The whole checkpoint is about 116 GiB, so it fits on one MI355X (288 GB HBM3E) with room for a large KV cache. That makes **TP1** the natural deployment: no tensor-parallel all-reduce in the decode path, and eight independent replicas per 8-GPU node.

The MTP draft is quantized like the target, so the draft's MoE runs on the same MXFP4 kernels.

## What we optimized

### Memory: the PLE table on the host

Loaded on the GPU, the PLE table takes a large share of the weight memory. SGLang's `--ple-offload-embedding` keeps it in CPU pinned memory in FP8 and gathers only the rows each step needs. GPU weight memory drops from about 130 GB to 82 GB, and the KV-cache pool grows from about 2.1M to 3.7M tokens. We also use the requested `--mem-fraction-static` as given (`SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1`) instead of the AITER backend's default 0.85 scaling for long contexts, which gives a 3.45M-token pool instead of 2.78M at 12–16 users.

### Low concurrency: the FlyDSL monokernel for speculative decoding

At 1–8 concurrent requests a decode step is dominated by kernel launches, not arithmetic. Each MTP step runs a target verify over every layer plus several draft steps, each made of many small kernels.

The **Qwen3.8-Flash-Next monokernel** is a set of persistent FlyDSL kernels that run the whole decoder stack of a target-verify step in **13 launches** and each MTP draft step in **one launch**. The kernels synchronize inside the GPU through mailboxes rather than returning to the host between operators. Because the monokernel's verify writes the per-draft intermediate GDN states itself, it does not need SGLang's ReplaySSM spec-verify path.

- It serves NEXTN with `eagle-topk 1` at up to 8 running requests (`SGLANG_ENABLE_QWEN4_MONOKERNEL=1`, `SGLANG_ENABLE_QWEN4_MTP_MONOKERNEL=1`).
- Making its guarded mailbox waits as cheap as unguarded ones removed a 19% step-time overhead at 4 requests and 6.9% at 1 ([sgl-amd-mono#19](https://github.com/sgl-project/sgl-amd-mono/pull/19)).
- The kernel is moving into [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono/pull/29) behind SGLang's monokernel interface (`--monokernel-backend amd`).

**[TODO: monokernel on vs off at c1/c2/c4/c8 on the AgentX traces — interactivity and throughput. Not measured yet.]**

### High concurrency: fused decode kernels

Above 8 users, SGLang's standard path with ReplaySSM spec-verify (`--enable-linear-replayssm-spec`) serves the decode steps, and the gains come from faster per-operator kernels:

- **QSA sparse decode** ([aiter#6670](https://github.com/ROCm/aiter/pull/6670), [sglang#45196](https://github.com/sgl-project/sglang/pull/45196)). The previous path copied each row's selected KV into scratch and then ran one attention program per (row, KV head), leaving most of the 256 compute units idle at decode batch sizes. One fused FlyDSL kernel now gathers the selected KV straight from the paged pool and splits each row across many workgroups. It is 3.4–5.7x faster per call (for example 59.7 vs 202.6 µs at 64 rows, the verify width at 16 users) and cuts QSA's share of decode GPU time from 9.4% to 2.8%. End to end at 32k/32k with 8 users: +10.0% output throughput, −9.3% time per output token.
- **Hyper-connection mix** ([aiter#5993](https://github.com/ROCm/aiter/pull/5993), [sglang#41880](https://github.com/sgl-project/sglang/pull/41880)). A FlyDSL `hc_mix` kernel fuses the low-rank gate's two GEMMs with their SiLU, sigmoid and stream-average epilogues: 18–26 µs per call against 36–46 µs for the Triton kernel at 1–16 rows (1.7–2.4x). On ROCm the split `hc_combine` pair is replaced by the single fused kernel, since both halves sit at the launch floor.
- **GDN conv1d and decode/verify** ([aiter#5992](https://github.com/ROCm/aiter/pull/5992), [sglang#41879](https://github.com/sgl-project/sglang/pull/41879)). FlyDSL kernels for the causal conv1d in GDN prefill and for the GDN recurrent decode and MTP verify (`SGLANG_AITER_GDN_CONV=1`, `SGLANG_AITER_GDN_DECODE=1`).
- **GEMM tuning** ([aiter#6769](https://github.com/ROCm/aiter/pull/6769)). Tuned gfx950 tables for the A8W8 bpreshuffle and BF16 GEMMs at Qwen3.8-Flash-Next's TP1 shapes, for example 54.4 → 41.3 µs and 16.8 → 13.6 µs for two hyper-connection GEMMs.

Together with the larger KV pool, these raised throughput at 16 agents from 69,285 to 76,943 tok/s per GPU (+11%) and median interactivity from 96.5 to 123.0 tok/s per user (+27%), with 18% lower mean inter-token latency. **[TODO: per-change ablation if we want one; today only the combined effect is measured, and it includes the memory-fraction change.]**

### Agentic serving configuration

Agent workloads use 16k prefill chunks, page size 64, BF16 Mamba/GDN state, six tokenizer workers, AITER unified attention, INT8 quick-reduce, streaming every 50 tokens, and the Qwen3 reasoning and Qwen3-Coder tool-call parsers. Admission and the decode CUDA-graph batch equal the number of agents at 1–8 users (monokernel) and twice the number of agents at 12–16 users, so bursts stay inside a captured graph. At 8 users the draft proposes one token per step instead of three.

## Performance

InferenceX AgentX agentic-coding benchmark: AIPerf replays recorded coding-agent sessions (mean input about 86k tokens, p90 about 218k) for one hour per point, on one MI355X (TP1).

| Agents | Total tok/s/GPU | Output tok/s/GPU | Interactivity p50 / p90 (tok/s/user) | TTFT p50 / p90 (s) | TPOT p50 (ms) | Decode path |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 16,613 | 181.0 | 407.1 / 397.4 | 0.33 / 1.43 | 2.46 | monokernel |
| 2 | 20,580 | 213.6 | 394.6 / 304.6 | 0.30 / 0.89 | 2.53 | monokernel |
| 4 | 25,223 | 283.2 | 322.5 / 231.1 | 0.29 / 0.74 | 3.10 | monokernel |
| 8 | 41,098 | 402.3 | 270.8 / 163.7 | 0.32 / 0.84 | 3.69 | monokernel, 1 draft token |
| 12 | 59,126 | 540.0 | 142.7 / 99.8 | 0.39 / 0.98 | 7.01 | ReplaySSM |
| 16 | 77,097 | 678.1 | 122.4 / 72.1 | 0.45 / 1.10 | 8.17 | ReplaySSM |

*Figure 1 [TODO: chart]: throughput per GPU against interactivity across 1–16 agents.*

Following the InferenceX AgentX rules, speculative decoding in these runs uses a fixed, committed acceptance length for this model (2.32 tokens per step at 3 draft steps, 1.65 at 1 step) so that results are comparable across submissions; accuracy is measured separately with real verification.

**[TODO: optionally a "what's coming" row — the full patch set, including PRs still in review, reached 81,208 tok/s/GPU at 151.4 tok/s/user p50 at 16 agents.]**

## How to run

The commands below use the public SGLang ROCm image on an MI355X node, one GPU per replica.

```bash
docker pull lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261026
docker run -it --device /dev/dri --device /dev/kfd \
  --network host --ipc host --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=64G -v /models:/models \
  lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261026
```

**[TODO: until the open PRs land, the image needs the patch set from the InferenceX recipe ([InferenceX#4542](https://github.com/SemiAnalysisAI/InferenceX/pull/4542), `qwen3.8next-rocm10-merged-patches.sh`). Decide whether to point readers there or wait for a release image.]**

**Recipe 1: 1–8 concurrent users (monokernel).** Set `--max-running-requests` and `--cuda-graph-max-bs-decode` to the number of users. At 8 users use `--speculative-num-steps 1 --speculative-num-draft-tokens 2`.

```bash
SGLANG_USE_AITER=1 SGLANG_USE_AITER_UNIFIED_ATTN=1 AITER_FLYDSL_FORCE=1 AITER_GDR_FLYDSL=1 \
ROCM_QUICK_REDUCE_QUANTIZATION=INT8 SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_AITER_GDN_CONV=1 SGLANG_AITER_GDN_DECODE=1 SGLANG_ENABLE_QSA_FLYDSL_DECODE=1 \
SGLANG_ENABLE_QWEN4_MONOKERNEL=1 SGLANG_ENABLE_QWEN4_MTP_MONOKERNEL=1 \
SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1 \
python3 -m sglang.launch_server \
  --model-path /models/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8 \
  --served-model-name qwen3.8-flash-next --trust-remote-code \
  --tensor-parallel-size 1 --quantization quark --attention-backend aiter \
  --page-size 64 --kv-cache-dtype auto --mem-fraction-static 0.9 \
  --chunked-prefill-size 16384 --mamba-ssm-dtype bfloat16 \
  --ple-offload-embedding --json-model-override-args '{"text_config": {"ple_embedding_dtype": "float8_e4m3fn"}}' \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --max-running-requests 4 --cuda-graph-max-bs-decode 4 \
  --tokenizer-worker-num 6 --scheduler-recv-interval 10 --stream-interval 50 \
  --model-loader-extra-config '{"enable_multithread_load": true}' --watchdog-timeout 1200 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --host 0.0.0.0 --port 8000
```

**Recipe 2: more than 8 concurrent users (ReplaySSM).** Drop the two monokernel variables, add `--enable-linear-replayssm-spec`, and set `--max-running-requests` and `--cuda-graph-max-bs-decode` to twice the number of users (32 for 16).

Quick check:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role": "user", "content": "What is 17*23? Answer briefly."}],
  "max_tokens": 200, "temperature": 0}'
```

## Accuracy

GSM8K, full 1,319-question test set, served with recipe 2 at 16 concurrent requests and real MTP verification: **0.973 strict match** (±0.004). The fused QSA decode kernel is within noise of the Triton path on GSM8K (short prompts 95.2% vs 96.0%; ~23k-token prompts 95.8% vs 97.9%) and closer to an fp64 reference (relative L2 error 1.7e-3 vs 4–6e-3).

**[TODO: AIME / LiveCodeBench / long-context needle results, if we want parity with the MiniMax post.]**

## What's next

- **Landing the remaining PRs.** The SGLang integrations ([#41879](https://github.com/sgl-project/sglang/pull/41879), [#41880](https://github.com/sgl-project/sglang/pull/41880), [#45196](https://github.com/sgl-project/sglang/pull/45196)) and the monokernel in sgl-amd-mono ([#29](https://github.com/sgl-project/sgl-amd-mono/pull/29)) are in review, so a future SGLang ROCm image will serve these recipes without patches.
- **More fused paths.** A larger set of changes still in review (fused QK-norm/RoPE/gate on HIP, fused GDN chunked prefill, AITER top-k gating for the 512-expert router, ratio-3 GDN packing) reached 81,208 tok/s/GPU and 151 tok/s per user at 16 agents in our tests.
- **Wider monokernel steps.** The monokernel stops at 8 requests; extending it to 12–16 would bring its launch savings to the high-concurrency points.

## Acknowledgements

We thank the Qwen team, the SGLang community, and the AMD AITER, FlyDSL, Quark, ROCm and InferenceX contributors whose work this post builds on. [TODO: names.]

## Test configuration and disclaimer

Measured on one GPU of an 8x AMD Instinct MI355X node in October 2026 with `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261026` (SGLang `5ef99013`, AITER `285579b6`), patched with the open PRs listed above, through the InferenceX AgentX agentic-coding harness (AIPerf replay of recorded coding-agent traces, 3,600 s per point, InferenceX run [40420614823](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40420614823)). Speculative-decoding throughput uses InferenceX's committed synthetic acceptance length; accuracy uses lm-eval with real verification. Results on other hardware, software versions, settings or workloads will differ; these figures are for illustration and are not a performance guarantee.

## Appendix: environment variables

| Variable | Set to | Description |
|---|---|---|
| `SGLANG_USE_AITER` | 1 | AITER kernels for MoE, attention and GEMMs |
| `SGLANG_USE_AITER_UNIFIED_ATTN` | 1 | AITER unified attention |
| `AITER_FLYDSL_FORCE` | 1 | prefer FlyDSL kernels where AITER has them |
| `AITER_GDR_FLYDSL` | 1 | FlyDSL kernels for the GDN recurrence |
| `ROCM_QUICK_REDUCE_QUANTIZATION` | INT8 | quick-reduce quantization (no all-reduce at TP1; harmless) |
| `SGLANG_ENABLE_SPEC_V2` | 1 | SGLang's v2 speculative-decoding scheduler |
| `SGLANG_AITER_GDN_CONV` | 1 | FlyDSL causal conv1d for GDN prefill |
| `SGLANG_AITER_GDN_DECODE` | 1 | FlyDSL GDN decode and MTP verify |
| `SGLANG_ENABLE_QSA_FLYDSL_DECODE` | 1 | fused FlyDSL QSA sparse decode (default on) |
| `SGLANG_ENABLE_QWEN4_MONOKERNEL` | 1 (1–8 users) / unset | monokernel for MTP target verify |
| `SGLANG_ENABLE_QWEN4_MTP_MONOKERNEL` | 1 (1–8 users) / unset | monokernel for MTP draft steps |
| `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION` | 1 | use `--mem-fraction-static` as given instead of the 0.85 long-context scaling |

#### Article by

Authors: Chang Liu (@ChangLiu0709), Nikolai (@amd-nprotaso), and Christian Gilli (@amd-cgilli) — AMD.
