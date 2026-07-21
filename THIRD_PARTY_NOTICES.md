# Third Party Notices

## SGLang

- Source: https://github.com/sgl-project/sglang
- Upstream version: v0.5.12
- License: Apache License, Version 2.0
- Original copyright: Copyright 2023-2024 SGLang Team
- Hygon modifications: HCU platform adaptations, build scripts, dependency adjustments, performance optimizations, and documentation updates.

## vLLM

- Source: https://github.com/vllm-project/vllm
- License: Apache License, Version 2.0
- Usage: runtime, distributed execution, model loading, benchmarking, attention, quantization, and kernel compatibility code paths include vLLM-attributed implementations or interfaces.

## FlashInfer

- Source: https://github.com/flashinfer-ai/flashinfer
- License: Apache License, Version 2.0, with additional third-party components documented by the FlashInfer project.
- Usage: attention, normalization, sampling, communication fusion, and kernel helper code paths include FlashInfer-attributed implementations or interfaces.

## NVIDIA TensorRT-LLM

- Source: https://github.com/NVIDIA/TensorRT-LLM
- License: Apache License, Version 2.0, with additional third-party components documented by the TensorRT-LLM project.
- Usage: MoE top-k kernel logic includes TensorRT-LLM-attributed implementation lineage through vLLM references.

## NVIDIA Transformer Engine

- Source: https://github.com/NVIDIA/TransformerEngine
- License: Apache License, Version 2.0
- Usage: multimodal context-parallel attention communication includes Transformer Engine-attributed implementation lineage.

## Flash Linear Attention

- Source: https://github.com/fla-org/flash-linear-attention
- License: MIT License
- Usage: FLA attention and layer-normalization modules include Flash Linear Attention-attributed implementations.

## DeepEP

- Source: https://github.com/deepseek-ai/DeepEP
- License: MIT License
- Usage: expert-parallel MoE dispatch and related kernel utility code paths include DeepEP-attributed algorithms or implementation references.

## FastVideo

- Source: https://github.com/hao-ai-lab/FastVideo
- License: Apache License, Version 2.0
- Usage: multimodal generation configuration, runtime, loading, platform, and pipeline components include FastVideo-attributed implementations.

## Hugging Face Diffusers

- Source: https://github.com/huggingface/diffusers
- License: Apache License, Version 2.0
- Usage: diffusion scheduler and pipeline helper code paths include Diffusers-attributed implementations.

## OpenAI Simple Evals

- Source: https://github.com/openai/simple-evals
- License: MIT License
- Usage: simple evaluation helpers include Simple Evals-attributed implementations.

## OpenAI HumanEval

- Source: https://github.com/openai/human-eval
- License: MIT License
- Usage: the restricted HCU HumanEval correctness judge adapts HumanEval's execution guard and per-sample correctness-check structure.

## OmniServe / QServe

- Source: https://github.com/mit-han-lab/omniserve
- License: Apache License, Version 2.0
- Usage: QServe W4A8 GEMM kernels and related Python exports include OmniServe/QServe-attributed implementations or interfaces.

## Batch Invariant Ops

- Source: https://github.com/thinking-machines-lab/batch_invariant_ops
- License: MIT License
- Usage: batch-invariant operation tests and integration points include Batch Invariant Ops-attributed implementations or interfaces.

## Mamba

- Source: https://github.com/state-spaces/mamba
- License: Apache License, Version 2.0
- Usage: Mamba-related registered tests and kernels include Mamba-attributed test or reference logic.
