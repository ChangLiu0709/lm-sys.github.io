---
title: "Qwen3.8-Flash-Next on AMD Instinct MI355X: serving and optimization with SGLang"
author: "Nikolai (@amd-nprotaso), Christian Gilli (@amd-cgilli), and Chang Liu (@ChangLiu0709) — AMD"
date: "November 18, 2026"
previewImg: /images/blog/qwen38-flash-next-mi355x/preview.png
type: blog
---

Qwen3.8-Flash-Next is the Qwen team's first open-weight release of the architecture that will underpin Qwen4: a 125B-parameter Mixture-of-Experts model that activates only 6B parameters per token, plus a 51B-parameter n-gram embedding table and a 4B multi-token-prediction (MTP) layer, with a native context of 262,144 tokens (extensible to 1M). It combines Gated DeltaNet linear attention with Qwen Sparse Attention (QSA), a block-level sparse attention that cuts long-context latency, and a Gated Residual that widens the residual stream into four branches. These choices target long-context agentic workloads, which increasingly dominate real-world usage, and its open weights, compatible with Hugging Face Transformers, vLLM and SGLang, give the community an early look at where the Qwen family is heading. Each of these features needs dedicated kernels and runtime support, so enabling and optimizing Qwen3.8-Flash-Next on AMD Instinct™ MI355X GPUs is important. In this blog, we show how to optimize its inference with fused FlyDSL kernels for QSA, the Gated Residual and Gated DeltaNet, a persistent monokernel for low-concurrency speculative decoding, and AI agents that automate part of the tuning. We benchmark the inference performance on InferenceX AgentX, which replays recorded coding-agent sessions (mean input of about 86k tokens) for one hour per concurrency level and reports throughput per GPU against per-user interactivity (tokens/s/user), and MI355X outperforms NVIDIA B200 and B300 across all concurrency levels.

This post describes how we serve Qwen3.8-Flash-Next on AMD Instinct MI355X GPUs with SGLang. We start with our latest performance results on InferenceX AgentX, then explain the optimizations behind them: the Qwen3.8-Flash-Next architecture, the key kernel optimizations, and how we use AI agents to automate part of the process. All results can be reproduced with the commands in the SGLang cookbook [link].

## Performance

We used InferenceX AgentX as our benchmark dataset and metrics, as [reasons]. As shown in Figure 1, [the overall configurations of the run, including concurrency list as ...] and [AgentX configurations, like including how many requests, benchmark time duration as 3600s]. [Analysis of the overall trend]. 

// screenshot from InferenceX

*Figure 1 [TODO: chart]: throughput per GPU against interactivity across 1–16 agents.*

[Analysis of the high concurrency scenario, with the optimizations, i.e., kernel fusion, xxx, (summary from our optimizations including the major software stacks)]. [Analysis of the low concurrency scenario: with monokernel, xxx]. We give more details about these optimizations in the later sections.

**NOTE:** As of November 19, 2026, the date of writing, the results shown in the figure depend on 7 PRs [PR links] that are awaiting merge into the SGLang main branch.

## Qwen3.8-Flash-Next Model Architecture

The architecture of Qwen3.8-Flash-Next is shown in Figure 2. Several of its properties matter for inference:

- **Gated DeltaNet.** 36 of the 48 layers use Gated DeltaNet, a linear-attention layer with a fixed-size recurrent state per request, so its decode cost does not grow with context.
- **Qwen Sparse Attention (QSA).** The other 12 use Qwen Sparse Attention: each query attends to a learned selection of past positions, with 24 query heads sharing 2 KV heads of dimension 256.
- **Hyper-connections (Gated Residual).** Instead of one residual stream, the model keeps four streams of width 2,560. Before every block, a gated mix collapses them into one input. It is small, but it runs in every layer at every step.
- **Native MTP.** A built-in draft layer proposes tokens for speculative decoding, which the target model verifies in one forward pass.

// diagram of Qwen3.8 Flash Next model architecture

### The checkpoint: MXFP4 experts with AMD Quark

For AMD Instinct GPUs, we quantized the Qwen3.8-Flash-Next weights with AMD Quark, which is [one sentence of Quark].
Then [the model can be served in a quantized form which runs more efficiently and reduces GPU memory consumption]. In our experiments, we serve [`amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8`](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8).

The whole checkpoint is about 116 GiB, so it fits on a single MI355X (288 GB HBM3E) with room left for a large KV cache, which makes ... [one sentence: serving more efficient]. The MTP draft model is quantized in the same way as the target model, so the draft's MoE runs on the same MXFP4 kernels.

## Kernel Optimization

### QSA - Qwen Sparse Attention
// analysis

// key change(s)

// performance gain

### Hyper Connections
// analysis

// key change(s)

// performance gain


### GDN
// analysis

// key change(s)

// performance gain


### Monokernel
// analysis

// key change(s)

// performance gain

### Optimization Automation
// key idea

// ...


## Conclusion


## Acknowledgements

We thank the Qwen team, the SGLang community, and the AMD AITER, FlyDSL, Quark, ROCm and InferenceX contributors whose work this post builds on. [TODO: names.]

## Test configuration and disclaimer

**Software stack of the recipes above:** `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261117` (SGLang `c67db7c762`, AITER `581579ab`, FlyDSL 0.4.2), patched with the open PRs listed above. **Numbers in Performance and Accuracy** were measured on one GPU of an 8x AMD Instinct MI355X node in October 2026 with `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261026` (SGLang `5ef99013`, AITER `285579b6`), patched with the PRs open at the time, through the InferenceX AgentX agentic-coding harness (AIPerf replay of recorded coding-agent traces, 3,600 s per point, InferenceX run [40420614823](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40420614823)). Per-PR numbers quoted in the optimization sections come from each pull request's own benchmarks, on the base and image given there. Speculative-decoding throughput uses InferenceX's committed synthetic acceptance length; accuracy uses lm-eval with real verification. Results on other hardware, software versions, settings or workloads will differ; these figures are for illustration and are not a performance guarantee.

#### Article by

Authors: Nikolai (@amd-nprotaso), Christian Gilli (@amd-cgilli), and Chang Liu (@ChangLiu0709) — AMD.
