---
title: "Qwen3.8-Flash-Next on AMD Instinct MI355X: serving and optimization with SGLang"
author: "Nikolai (@amd-nprotaso), Christian Gilli (@amd-cgilli), and Chang Liu (@ChangLiu0709) — AMD"
date: "November 18, 2026"
previewImg: /images/blog/qwen38-flash-next-mi355x/preview.png
type: blog
---

Qwen3.8-Flash-Next is [TODO: one-line model positioning from the Qwen team, e.g. "the Qwen team's efficient flagship for coding and agent workloads"]. [One sentence for its community impact, and the reason binding to its model architecture, features, dataset(if involved), etc]. Thus, enablement and performance optimization of Qwen3.8-Flash-Next on MI355X is important. In our blog, we show how to optimize the inference of Qwen3.8-Flash-Next by ... We benchmark the inference performance on InferenceX AgentX, which [one sentence to introduce the dataset and benchmark metrics], outperforming our competitor NVIDIA B200 and B300 across all concurrency levels.

This post describes how we serve Qwen3.8-Flash-Next on AMD Instinct MI355X GPUs with SGLang. We start with our latest performance result on InferenceX AgentX, then reveal the optimization behind the scene by briefly introducing the Qwen model architecure, the key optimizations, and the ideas of using AI agent to automate the process. All the results can be reproduced by referring to the command on SGLang cookbook [link].

## Performance

We used InferenceX AgentX as our benchmark dataset and metrics, as [reasons]. As shown in Figure 1, [the overall configurations of the run, including concurrency list as ...] and [AgentX configurations, like including how many requests, benchmark time duration as 3600s]. [Analysis of the overall trend]. 

// screenshot from InferenceX
*Figure 1 [TODO: chart]: throughput per GPU against interactivity across 1–16 agents.*

[Analysis of the high concurrency scenario, with the optimizations, i.e., kernel fusion, xxx, (summary from our optimizations including the major software stacks)]. [Analysis of the low concurrency scenario: with monokernel, xxx]. We are going to give more details about the optimizations in the latter sections.

**NOTE:** Until the day Nov 19, 2026, we wrote the blog, the performance submission shown in the figure depends on 7 PRs [PR links] that awaits to be merged to the SGLang main branch.

## Qwen3.8-Flash-Next Model Architecture

The model architecture of Qwen3.8-Flash-Next is shown in Figure 2. For its inference, five properties of the model architecture matter:

- **Hybrid attention.** 36 of the 48 layers use Gated DeltaNet, a linear-attention layer with a fixed-size recurrent state per request, so its decode cost does not grow with context. The other 12 use QuerySparse Attention: each query attends to a learned selection of past positions, with 24 query heads sharing 2 KV heads of dimension 256.
- **Hyper-connections.** Instead of one residual stream, the model keeps 4 streams of width 2,560. Before every block a gated mix collapses them into one input. It is small but runs in every layer at every step.
- **Sparse MoE.** Every layer is a sparse MoE: 512 routed experts with a softmax router, 10 routed experts active per token, plus one shared expert.
- **Per-layer embeddings (PLE).** A large n-gram embedding table feeds the model's PLE layer. It is read sparsely, which makes it a good candidate for host memory.
- **Native MTP.** A built-in draft layer proposes tokens for speculative decoding, verified by the target in one forward pass.

// diagram of Qwen3.8 Flash Next model architecture

### The checkpoint: MXFP4 experts with AMD Quark

On AMD MI GPUs, we quantized the Qwen3.8 Flash Next model weights using AMD Quark, which is [one sentence of Quark].
Then [the model can be served in a quantized form which runs more efficiently and reduce the comsuption of GPU memory]. In our experiments, we serve `[amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8)`:

| Component                             | Format                                            |
| ------------------------------------- | ------------------------------------------------- |
| Routed experts (target and MTP draft) | OCP MXFP4 (FP4 with one E8M0 scale per 32 values) |
| Attention projections                 | FP8, per-channel scales                           |
| PLE embedding table                   | FP8 (`float8_e4m3fn`)                             |
| Router, norms, LM head                | High precision                                    |

The whole checkpoint is about 116 GiB, so it reduced the comsuption on one MI355X (288 GB HBM3E) with room for a large KV cache, which makes ... [once sentence: serving more efficient]. The MTP draft model is also quantized like the way we quantized the target model, so the draft's MoE runs on the same MXFP4 kernels.

## Kernel and Runtime Optimizations on MI355X





## Reproduction Commands


## Conclusion


## Acknowledgements

We thank the Qwen team, the SGLang community, and the AMD AITER, FlyDSL, Quark, ROCm and InferenceX contributors whose work this post builds on. [TODO: names.]

## Test configuration and disclaimer

**Software stack of the recipes above:** `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261117` (SGLang `c67db7c762`, AITER `581579ab`, FlyDSL 0.4.2), patched with the open PRs listed above. **Numbers in Performance and Accuracy** were measured on one GPU of an 8x AMD Instinct MI355X node in October 2026 with `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261026` (SGLang `5ef99013`, AITER `285579b6`), patched with the PRs open at the time, through the InferenceX AgentX agentic-coding harness (AIPerf replay of recorded coding-agent traces, 3,600 s per point, InferenceX run [40420614823](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40420614823)). Per-PR numbers quoted in the optimization sections come from each pull request's own benchmarks, on the base and image given there. Speculative-decoding throughput uses InferenceX's committed synthetic acceptance length; accuracy uses lm-eval with real verification. Results on other hardware, software versions, settings or workloads will differ; these figures are for illustration and are not a performance guarantee.

#### Article by

Authors: Nikolai (@amd-nprotaso), Christian Gilli (@amd-cgilli), and Chang Liu (@ChangLiu0709) — AMD.
