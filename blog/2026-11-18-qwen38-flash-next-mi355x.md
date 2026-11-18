---
title: "Qwen3.8-Flash-Next on AMD Instinct MI355X: serving and optimization with SGLang"
author: "Nikolai (@amd-nprotaso), Christian Gilli (@amd-cgilli), and Chang Liu (@ChangLiu0709) — AMD"
date: "November 18, 2026"
previewImg: /images/blog/qwen38-flash-next-mi355x/preview.png
type: blog
---

*Draft, 2026-11-18. The technical sections (what we optimized, how to run, environment variables, test configuration) were refreshed against the `20261117` image and the current patch set. The numbers in Performance and Accuracy are from 2026-10-28 and still need to be re-measured. Items marked **[TODO]** need a measurement or a decision before publication.*

## TL;DR

- Qwen3.8-Flash-Next, a hybrid linear/sparse-attention Mixture-of-Experts model with native multi-token prediction (MTP), runs on a **single** AMD Instinct™ MI355X GPU with SGLang. The model's 36 Gated DeltaNet (GDN) layers, 12 QuerySparse Attention (QSA) layers, hyper-connections, 512-expert MoE and the MTP draft all run on AITER and FlyDSL kernels.
- An AMD Quark checkpoint with MXFP4 experts (including the MTP draft), per-channel FP8 attention and an FP8 per-layer-embedding (PLE) table fits in about 116 GiB. Offloading the PLE table to CPU pinned memory cuts GPU weight memory from about 130 GB to 82 GB and grows the KV-cache pool from about 2.1M to 3.7M tokens. At 12 or more agents, an FP8 KV cache for the target and the draft, and at 48–80 agents HiCache host-memory offload, keep long agent contexts resident.
- At 1–8 concurrent users, the FlyDSL **monokernel** (in [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono), `--enable-amd-monokernel`) runs the whole MTP target-verify step in 13 persistent launches and each draft step in one launch, instead of hundreds of small kernels.
- Fused FlyDSL kernels replace the hot decode and prefill operators: QSA sparse decode (120 → 33 µs per call, +10% output throughput at 32k/32k with 8 users), a fused hyper-connection norm, mix and inject (1.4–3.0x per site), GDN conv1d and a single-launch GDN chunked prefill (1 launch instead of 8), and a top-k-only target-probability step for MTP verify (3.3 → 1.2 ms of eager time per decode step).
- On the InferenceX AgentX agentic-coding benchmark (real coding-agent traces, one hour per point), one MI355X served **16,613 total (input + output) tok/s at 407 tok/s per user** for one agent, and **77,097 tok/s at 122 tok/s per user** for 16 agents in our October run. GSM8K is 97.3% (strict match). **[TODO: refresh with the current recipe, including c24/c48/c80.]**
- Most of the stack is already upstream: the AITER kernels ship in the public `lmsysorg/sglang-rocm` image (AITER `581579ab`), and so do the monokernel, the FlyDSL hyper-connection mix, the AITER top-k router and the ratio-3 GDN packing. Seven SGLang pull requests are still in review and are applied as small source patches on top of the public image; no custom image is needed.

## Introduction

Qwen3.8-Flash-Next is [TODO: one-line model positioning from the Qwen team, e.g. "the Qwen team's efficient flagship for coding and agent workloads"]. Its architecture combines three ideas that each need their own kernels: linear attention through Gated DeltaNet for most layers, a learned sparse attention (QSA) for the rest, and hyper-connections that carry four copies of the hidden state between layers. Agentic coding adds a fourth requirement: very long, mostly shared prompts (tens to hundreds of thousands of tokens) with short, latency-sensitive replies.

This post describes how we serve Qwen3.8-Flash-Next on AMD Instinct MI355X GPUs with SGLang. We start with the measured performance on the InferenceX AgentX benchmark, then cover the model, the quantized checkpoint, the kernels we wrote or tuned in AITER and FlyDSL, the FlyDSL monokernel for speculative decoding at low concurrency, and a serving recipe.

## Performance

InferenceX AgentX agentic-coding benchmark: AIPerf replays recorded coding-agent sessions (mean input about 86k tokens, p90 about 218k) for one hour per point, on one MI355X (TP1). The last column names the decode path; the monokernel and ReplaySSM are explained in [What we optimized](#what-we-optimized).

| Agents | Total tok/s/GPU | Output tok/s/GPU | Interactivity p50 / p90 (tok/s/user) | TTFT p50 / p90 (s) | TPOT p50 (ms) | Decode path |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 16,613 | 181.0 | 407.1 / 397.4 | 0.33 / 1.43 | 2.46 | monokernel |
| 2 | 20,580 | 213.6 | 394.6 / 304.6 | 0.30 / 0.89 | 2.53 | monokernel |
| 4 | 25,223 | 283.2 | 322.5 / 231.1 | 0.29 / 0.74 | 3.10 | monokernel |
| 8 | 41,098 | 402.3 | 270.8 / 163.7 | 0.32 / 0.84 | 3.69 | monokernel, 1 draft token |
| 12 | 59,126 | 540.0 | 142.7 / 99.8 | 0.39 / 0.98 | 7.01 | ReplaySSM |
| 16 | 77,097 | 678.1 | 122.4 / 72.1 | 0.45 / 1.10 | 8.17 | ReplaySSM |

*Figure 1 [TODO: chart]: throughput per GPU against interactivity across 1–16 agents.*

These rows were measured on 2026-10-28 with the image and patch set of that date (see Test configuration). The 8-agent row used one draft token per step; the monokernel in the current image accepts only 3 draft steps, so that point needs a re-run. **[TODO: replace the table with the current recipe's results at 1, 2, 12, 24, 48 and 80 agents.]**

Following the InferenceX AgentX rules, speculative decoding in these runs uses a fixed, committed acceptance length for this model (2.32 tokens per step at 3 draft steps, 1.65 at 1 step) so that results are comparable across submissions; accuracy is measured separately with real verification.

**[TODO: optionally a "what's coming" row. In October the full patch set, including the PRs still in review, reached 81,208 tok/s/GPU at 151.4 tok/s/user p50 at 16 agents; re-measure it on the current stack.]**

## Qwen3.8-Flash-Next at a glance

For inference, five properties matter:

- **Hybrid attention.** 36 of the 48 layers use Gated DeltaNet, a linear-attention layer with a fixed-size recurrent state per request, so its decode cost does not grow with context. The other 12 use QuerySparse Attention: each query attends to a learned selection of at most 2,051 past positions (512 compressed blocks of 4, plus 3), with 24 query heads sharing 2 KV heads of dimension 256.
- **Hyper-connections.** Instead of one residual stream, the model keeps 4 streams of width 2,560. Before every block a gated mix (a 10,240 → 320 → 10,240 low-rank gate) collapses them into one input. It is small but runs in every layer at every step.
- **Sparse MoE.** Every layer is a sparse MoE: 512 routed experts with a softmax router, 10 routed experts active per token, plus one shared expert (fused into the routed MXFP4 experts as expert 512). **[TODO: total and active parameter counts.]**
- **Per-layer embeddings (PLE).** A large n-gram embedding table feeds the model's PLE layer (layer 1 in this checkpoint). It is read sparsely, which makes it a good candidate for host memory.
- **Native MTP.** A built-in draft layer proposes tokens for speculative decoding (SGLang `NEXTN`), verified by the target in one forward pass.

### The checkpoint: MXFP4 experts with AMD Quark

The model is served in a quantized form that fits on one GPU. We serve [`amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8`](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8), produced with AMD Quark:

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

Loaded on the GPU, the PLE table takes a large share of the weight memory. Outside the monokernel points, SGLang's `--ple-offload-embedding` keeps it in CPU pinned memory in FP8 and gathers only the rows each step needs. GPU weight memory drops from about 130 GB to 82 GB, and the KV-cache pool grows from about 2.1M to 3.7M tokens.

### Low concurrency: the FlyDSL monokernel for speculative decoding

At 1–8 concurrent requests a decode step is dominated by kernel launches, not arithmetic. Each MTP step runs a target verify over every layer plus several draft steps, each made of many small kernels.

The **Qwen3.8-Flash-Next monokernel** is a set of persistent FlyDSL kernels that run the whole model for a target-verify step in **13 launches** (layer 0 behind the prelude of token embedding and PLE history, layers 1–3 behind layer 1's PLE, then 11 launches of 3 GDN layers and 1 QSA layer each, the last followed by the model-level hyper-connection mixer) and each MTP draft step in **one launch**. The kernels synchronize inside the GPU through mailboxes rather than returning to the host between operators, and they read SGLang's weights in place. The monokernel's verify writes the per-draft intermediate GDN states itself, so it does not use SGLang's ReplaySSM spec-verify path.

- It lives in [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono/pull/29) behind SGLang's monokernel interface and is in the public ROCm image. Turn it on with `--enable-amd-monokernel`; SGLang falls back to its own forward for any step the kernel does not take (for example a batch above 8 requests).
- It serves NEXTN target verify of 1–8 requests × 4 tokens, one draft chain of 3 steps (`--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`), with TP1, a Quark checkpoint, and a BF16 KV cache (`--kv-cache-dtype auto`). It needs ReplaySSM off, and the PLE table on the GPU (`--ple-offload-embedding` off), because its prelude gathers the n-gram table on the device.
- Making its guarded mailbox waits as cheap as unguarded ones removed a 19% step-time overhead at 4 requests and 6.9% at 1 ([sgl-amd-mono#19](https://github.com/sgl-project/sgl-amd-mono/pull/19)).

**[TODO: monokernel on vs off at c1/c2/c4/c8 on the AgentX traces — interactivity and throughput. Not measured yet.]**

### High concurrency: fused decode and prefill kernels

Above 8 users, SGLang's standard path with ReplaySSM spec-verify (`--enable-linear-replayssm-spec`) serves the decode steps, and the gains come from faster per-operator kernels. The AITER kernels below are merged and in the image; the SGLang changes that call them are either merged or in review, as noted.

- **QSA sparse decode** ([aiter#6670](https://github.com/ROCm/aiter/pull/6670) merged, [sglang#45196](https://github.com/sgl-project/sglang/pull/45196) in review). The previous path copied each row's selected KV into scratch and then ran one attention program per (row, KV head), leaving most of the 256 compute units idle at decode batch sizes. One fused FlyDSL kernel now gathers the selected KV straight from the paged pool and splits each row across many workgroups. It is 3.4–5.7x faster per call (for example 59.7 vs 202.6 µs at 64 rows, the verify width at 16 users) and cuts QSA's share of decode GPU time from 9.4% to 2.8%. End to end at 32k/32k with 8 users: +10.0% output throughput, −9.3% time per output token. It is on whenever `SGLANG_USE_AITER=1` on gfx950.
- **QSA indexer logits** ([sglang#48649](https://github.com/sgl-project/sglang/pull/48649), in review). The indexer's decode MQA-logits kernel gets the query padded to the 16-head MFMA width once, and each workgroup scores four consecutive 64-key groups, with the grid spanning the page table. On mixed-length batches the kernel drops from 243 to 89 µs per call, and it runs 13 times per decode step. Logits and top-k are bit-exact against the previous launch.
- **Hyper-connections** ([aiter#5993](https://github.com/ROCm/aiter/pull/5993) and [sglang#41880](https://github.com/sgl-project/sglang/pull/41880), both merged; [sglang#48648](https://github.com/sgl-project/sglang/pull/48648), in review). The FlyDSL `hc_mix` kernel fuses the low-rank gate's two GEMMs with their SiLU, sigmoid and stream-average epilogues: 18–26 µs per call against 36–46 µs for the Triton kernel at 1–16 rows (1.7–2.4x). #48648 goes further and folds the per-branch RMSNorm and the inject GEMV into the mix, so a hyper-connection site (mix and combine) at decode takes 12.7 µs instead of 38.2 µs at 1 row, 17.1 instead of 34.3 µs at 24 rows, and 29.3 instead of 41.6 µs at 128 rows. In the PR's AgentX run at 24 agents it gave +5.6% throughput and +14.9% P90 interactivity. It is off by default (`SGLANG_ROCM_OPT_QWEN4_FUSED_HC=1`) and covers batches of up to 128 rows.
- **GDN conv1d and decode/verify** ([aiter#5992](https://github.com/ROCm/aiter/pull/5992) merged, [sglang#41879](https://github.com/sgl-project/sglang/pull/41879) in review). FlyDSL kernels for the causal conv1d in GDN prefill (`SGLANG_AITER_GDN_CONV=1`) and for the GDN recurrent decode and MTP verify (`AITER_GDR_FLYDSL=1`).
- **GDN fused chunked prefill** ([aiter#6496](https://github.com/ROCm/aiter/pull/6496) merged, [sglang#39595](https://github.com/sgl-project/sglang/pull/39595) and [sglang#44103](https://github.com/sgl-project/sglang/pull/44103) in review). The previous prefill path took 8 launches per GDN layer and wrote every per-chunk state to HBM and read it back. One FlyDSL kernel now runs the l2norm, the prepare stage, the state recurrence and the output, keeps the state on chip, and writes the radix-cache snapshot itself: for a 16K-token chunk, 1 launch and 797 µs instead of 8 launches and 1,267–1,299 µs. For one GDN layer at 32K tokens, it takes 1.54 ms against 3.37 ms for Triton. It needs a BF16 SSM state (`--mamba-ssm-dtype bfloat16`).
- **MTP verify sampling** ([sglang#48090](https://github.com/sgl-project/sglang/pull/48090), in review). Qwen3.8 ships `top_k=20`, so every request samples from a handful of tokens, yet verify sorted the whole 248k-token vocabulary. The new path builds the target probabilities on each row's top-k support only: 2.74 → 0.30 ms at 64 rows, and 3.34 → 1.23 ms of eager GPU time per decode step. Batches with a `top_k` outside 2–128 keep the dense path. In the PR's AgentX runs it gave +7.3% median interactivity at 16 agents and +4.1% at 12, with GSM8K unchanged (95.8% both).
- **Routing, packing and GEMM tuning** (merged). AITER top-k gating for the 512-expert router ([sglang#40708](https://github.com/sgl-project/sglang/pull/40708)), fused GDN packing for ratio-3 heads ([sglang#44302](https://github.com/sgl-project/sglang/pull/44302)), strided GDN prefill projection views ([sglang#41683](https://github.com/sgl-project/sglang/pull/41683)), and tuned gfx950 tables for the A8W8 bpreshuffle and BF16 GEMMs at Qwen3.8-Flash-Next's TP1 shapes ([aiter#6769](https://github.com/ROCm/aiter/pull/6769), for example 54.4 → 41.3 µs and 16.8 → 13.6 µs for two hyper-connection GEMMs).

In our October run, the then-current set of these changes, together with the larger KV pool, raised throughput at 16 agents from 69,285 to 76,943 tok/s per GPU (+11%) and median interactivity from 96.5 to 123.0 tok/s per user (+27%), with 18% lower mean inter-token latency. **[TODO: per-change ablation if we want one; the individual PR numbers above come from each PR's own runs on different bases and should not be added up.]**

### Long agent contexts: FP8 KV cache, Mamba state pool and HiCache

From 12 agents up, the working set is many long, mostly shared prompts, so we change how the cache is stored:

- **FP8 KV cache** for the target and the draft (`--kv-cache-dtype fp8_e4m3 --speculative-draft-kv-cache-dtype fp8_e4m3`), with `--mem-fraction-static 0.88`. **[TODO: pool size and accuracy impact.]**
- **Fewer Mamba checkpoints.** The Mamba/GDN radix cache uses `--mamba-radix-cache-strategy extra_buffer_lazy` with `--mamba-track-interval 262144`, so a prompt gets no interior checkpoints, and `--max-mamba-cache-size` is set per point (960 at 12–24 agents, 720 at 48, 640 at 80).
- **HiCache** at 48 and 80 agents (`--enable-hierarchical-cache --hicache-size 298 --hicache-io-backend kernel --hicache-mem-layout page_first --hicache-write-policy write_back`): the KV cache spills to about 298 GB of host DRAM per replica, and admission goes up to 128 requests. At 80 agents we also cache only the prompt on finish (`--strip-thinking-cache`, since the AgentX replay never re-sends model output) and keep at most two Mamba states per conversation path (`--mamba-max-states-per-path 2`).
- **Breakable prefill CUDA graphs** (`--cuda-graph-backend-prefill breakable`) capture the GDN prefill on static per-bucket tables; a batch outside the captured buckets runs eagerly through the AITER fused prefill.

### Agentic serving configuration

Agent workloads use 16k prefill chunks, page size 64, BF16 Mamba/GDN state, six tokenizer workers, AITER unified attention, streaming every 50 tokens, and the Qwen3 reasoning and Qwen3-Coder tool-call parsers. Admission and the decode CUDA-graph batch equal the number of agents at 1–2 users (monokernel) and twice the number of agents at 12–24 users, so bursts stay inside a captured graph; at 48 and 80 agents both are 128. Every point uses `--mem-fraction-static` as given (`SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1`) instead of the AITER backend's default 0.85 scaling for long contexts, which gives a 3.45M-token pool instead of 2.78M at 12–16 users.

## How to run

The commands below use the public SGLang ROCm image on an MI355X node, one GPU per replica. The image already contains the AITER FlyDSL kernels (AITER `581579ab`, FlyDSL 0.4.2), the monokernel and the merged SGLang changes.

```bash
docker pull lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261117
docker run -it --device /dev/dri --device /dev/kfd \
  --network host --ipc host --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=64G -v /models:/models \
  lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261117
```

**[TODO: until the open PRs land, the image needs seven SGLang source patches (sglang#48648, #45196, #48649, #45529, #41879, #44103, #48090), kept in the InferenceX recipe ([InferenceX#4832](https://github.com/SemiAnalysisAI/InferenceX/pull/4832), `qwen3.8next/sglang/mi355x-fp4-mtp/patches/`, applied by `qwen3.8next-rocm10-patches.sh`; AITER is not touched). Decide whether to point readers there or wait for a release image. Recipes 1–3 run without the patches too, with the kernels they add left off.]**

**Recipe 1: 1–8 concurrent users (monokernel).** Set `--max-running-requests` and `--cuda-graph-max-bs-decode` to the number of users (up to 8). Keep NEXTN at 3 steps and 4 draft tokens: the monokernel serves only that shape. ReplaySSM, the PLE offload and the FP8 KV cache stay off.

```bash
SGLANG_USE_AITER=1 AITER_GDR_FLYDSL=1 SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_AITER_GDN_CONV=1 SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1 \
python3 -m sglang.launch_server \
  --model-path /models/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8 \
  --served-model-name qwen3.8-flash-next --trust-remote-code \
  --tensor-parallel-size 1 --quantization quark --attention-backend aiter \
  --page-size 64 --kv-cache-dtype auto --mem-fraction-static 0.9 \
  --chunked-prefill-size 16384 --mamba-ssm-dtype bfloat16 \
  --json-model-override-args '{"text_config": {"ple_embedding_dtype": "float8_e4m3fn"}}' \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-amd-monokernel \
  --max-running-requests 2 --cuda-graph-max-bs-decode 2 \
  --tokenizer-worker-num 6 --scheduler-recv-interval 10 --stream-interval 50 \
  --model-loader-extra-config '{"enable_multithread_load": true}' --watchdog-timeout 1200 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --host 0.0.0.0 --port 8000
```

**Recipe 2: 12–24 concurrent users (ReplaySSM, FP8 KV cache).** Replace the monokernel flag with the ReplaySSM and cache options below, turn the PLE offload on, and set `--max-running-requests` and `--cuda-graph-max-bs-decode` to twice the number of users (48 for 24). Add `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `SGLANG_AITER_GDN_PREFILL=1`, `SGLANG_AITER_GDN_PREFILL_H=fused` and `SGLANG_ROCM_OPT_QWEN4_FUSED_HC=1` to the environment (the last three come from the open PRs).

```bash
  --ple-offload-embedding --enable-linear-replayssm-spec \
  --kv-cache-dtype fp8_e4m3 --speculative-draft-kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.88 \
  --max-mamba-cache-size 960 --mamba-radix-cache-strategy extra_buffer_lazy --mamba-track-interval 262144 \
  --cuda-graph-backend-prefill breakable \
  --max-running-requests 48 --cuda-graph-max-bs-decode 48
```

**Recipe 3: 48–80 concurrent users (HiCache).** Start from recipe 2, raise both limits to 128, set `--max-mamba-cache-size` to 720 (48 users) or 640 (80 users), and add HiCache. At 80 users also add `--strip-thinking-cache --mamba-max-states-per-path 2`.

```bash
  --enable-hierarchical-cache --hicache-size 298 --hicache-io-backend kernel \
  --hicache-mem-layout page_first --hicache-write-policy write_back
```

Quick check:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role": "user", "content": "What is 17*23? Answer briefly."}],
  "max_tokens": 200, "temperature": 0}'
```

## Accuracy

GSM8K, full 1,319-question test set, served with recipe 2 at 16 concurrent requests and real MTP verification: **0.973 strict match** (±0.004). The fused QSA decode kernel is within noise of the Triton path on GSM8K (short prompts 95.2% vs 96.0%; ~23k-token prompts 95.8% vs 97.9%) and closer to an fp64 reference (relative L2 error 1.7e-3 vs 4–6e-3).

The newer changes were each checked on GSM8K with real MTP verification in their own pull requests: 96.3% with the fused hyper-connection norm on against 96.6% off ([sglang#48648](https://github.com/sgl-project/sglang/pull/48648)), 96.7% with the fused GDN chunked prefill and 96.9–97.0% with the staged one ([sglang#44103](https://github.com/sgl-project/sglang/pull/44103)), 95.8% for the top-k-only verify sampling and for main ([sglang#48090](https://github.com/sgl-project/sglang/pull/48090)), and 96.8% for the QSA indexer change on the full stack ([sglang#48649](https://github.com/sgl-project/sglang/pull/48649)). **[TODO: one GSM8K run of the final recipe, plus AIME / LiveCodeBench / long-context needle results, if we want parity with the MiniMax post.]**

## AMD Software Stacks

Everything in this post runs on the public `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261117` image, plus the SGLang source patches described above. The AMD pieces of that stack:

| Component | Version in the image | Role for Qwen3.8-Flash-Next |
|---|---|---|
| [ROCm](https://rocm.docs.amd.com/) | 10.0.0 | GPU runtime and compiler for MI355X (gfx950) |
| PyTorch | 2.11.0+rocm10.0.0 | tensors, CUDA-graph capture and the eager fallback paths |
| Triton | 3.8.0 (ROCm build) | SGLang's remaining Triton kernels, such as the QSA bookkeeping and the dense sampling fallback |
| [AITER](https://github.com/ROCm/aiter) | 0.1.29.dev156 (`581579ab`, 2026-11-07) | MXFP4 MoE, FP8 and BF16 GEMMs with tuned gfx950 tables, unified attention, top-k gating, and the FlyDSL kernels for GDN, QSA and hyper-connections |
| [FlyDSL](https://github.com/ROCm/FlyDSL) | 0.4.2 | the kernel language behind the fused GDN, QSA and hyper-connection kernels and the monokernel |
| [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) | `a04ceab` (2026-11-16) | the Qwen3.8-Flash-Next monokernel for MTP target verify and draft steps |
| [SGLang](https://github.com/sgl-project/sglang) | 0.5.23.dev20261117 (`c67db7c762`) | serving: scheduler, speculative decoding (NEXTN), ReplaySSM, radix cache and HiCache |
| [AMD Quark](https://quark.docs.amd.com/) | used offline | quantizes the checkpoint to MXFP4 experts, FP8 attention and an FP8 PLE table |

Two things in this stack are worth knowing when you reproduce the results:

- **The AITER pin carries the kernels.** SGLang's AITER pin was moved to `581579ab` in [sglang#50759](https://github.com/sgl-project/sglang/pull/50759), which contains aiter#5992, #5993, #6496, #6670 and #6769. Earlier images (for example `20261113`) still had the October AITER and none of these kernels, so use `20261117` or later.
- **The monokernel and the FlyDSL kernels are optional at runtime.** SGLang falls back to its own path when a kernel does not accept a step or when FlyDSL is not importable, so the server still starts, only slower. (The InferenceX patch script is stricter and fails the worker if a FlyDSL module is missing, so a benchmark never runs on the slow path by accident.)

**[TODO: Quark version used for the checkpoint, and the minimum driver / firmware versions we tested.]**

## Acknowledgements

We thank the Qwen team, the SGLang community, and the AMD AITER, FlyDSL, Quark, ROCm and InferenceX contributors whose work this post builds on. [TODO: names.]

## Test configuration and disclaimer

**Software stack of the recipes above:** `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261117` (SGLang `c67db7c762`, AITER `581579ab`, FlyDSL 0.4.2), patched with the open PRs listed above. **Numbers in Performance and Accuracy** were measured on one GPU of an 8x AMD Instinct MI355X node in October 2026 with `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261026` (SGLang `5ef99013`, AITER `285579b6`), patched with the PRs open at the time, through the InferenceX AgentX agentic-coding harness (AIPerf replay of recorded coding-agent traces, 3,600 s per point, InferenceX run [40420614823](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40420614823)). Per-PR numbers quoted in the optimization sections come from each pull request's own benchmarks, on the base and image given there. Speculative-decoding throughput uses InferenceX's committed synthetic acceptance length; accuracy uses lm-eval with real verification. Results on other hardware, software versions, settings or workloads will differ; these figures are for illustration and are not a performance guarantee.

## Appendix: environment variables

| Variable | Set to | Description |
|---|---|---|
| `SGLANG_USE_AITER` | 1 | AITER kernels for MoE, attention and GEMMs; also turns on the fused QSA sparse decode and the top-k MTP verify sampling on gfx950 |
| `SGLANG_USE_AITER_UNIFIED_ATTN` | 1 (12+ users) | AITER unified attention |
| `AITER_GDR_FLYDSL` | 1 | FlyDSL kernels for the GDN recurrence (decode and MTP verify) |
| `SGLANG_ENABLE_SPEC_V2` | 1 | SGLang's v2 speculative-decoding scheduler |
| `SGLANG_AITER_GDN_CONV` | 1 | FlyDSL causal conv1d for GDN prefill (sglang#41879) |
| `SGLANG_AITER_GDN_PREFILL` | 1 (12+ users) | AITER GDN chunked prefill (sglang#44103) |
| `SGLANG_AITER_GDN_PREFILL_H` | `fused` (12+ users) | single-launch fused FlyDSL GDN prefill; needs `--mamba-ssm-dtype bfloat16` |
| `SGLANG_ROCM_OPT_QWEN4_FUSED_HC` | 1 (12+ users) | fused hyper-connection norm, mix and inject at decode (sglang#48648) |
| `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION` | 1 | use `--mem-fraction-static` as given instead of the 0.85 long-context scaling |

The monokernel has no environment variable: pass `--enable-amd-monokernel`. **[TODO: sglang#44103 is being reworked into the `--linear-attn-prefill-backend flydsl_fused` backend, so the two GDN prefill variables above will change; the recipe branch still sets `SGLANG_OPT_QWEN4_FUSED_HC`, the name before the review rename.]**

#### Article by

Authors: Nikolai (@amd-nprotaso), Christian Gilli (@amd-cgilli), and Chang Liu (@ChangLiu0709) — AMD.
