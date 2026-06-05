# Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation

[![arXiv](https://img.shields.io/badge/arXiv-2510.22067-b31b1b.svg)](https://arxiv.org/abs/2510.22067)
[![Conference](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**Published at ICML 2026.**

This repository contains the implementation of the proposed method, GIFT (Gaze Shift-Guided Cross-modal Fusion Enhancement), that mitigates hallucinations in Vision-Language Models (VLMs).

The implementation builds upon Transformers v4.50.0 by modifying attention computation in QwenVL2 and Llama architectures. Visual saliency maps are integrated into the attention mechanism to enhance cross-modal fusion.

## Abstract

Vision language models (VLMs) often generate hallucination, i.e., content that cannot be substantiated by either textual or visual inputs. Prior work primarily attributes this to over-reliance on linguistic prior knowledge rather than visual inputs. Some methods attempt to mitigate hallucination by amplifying visual token attention proportionally to their attention scores. However, these methods overlook the visual attention sink problem, where attention is frequently misallocated to task-irrelevant visual regions, and neglect cross-modal fusion balance by enhancing only visual attention without adjusting attention to the user query. This can result in amplifying incorrect areas while failing to properly interpret the user query. To address these challenges, we propose a simple yet effective method called Gaze Shift-Guided Cross-modal Fusion Enhancement (GIFT). GIFT pre-computes a holistic visual saliency map by tracking positive changes in visual attention, or "gaze shifts", during user query comprehension, and leverages this map to amplify attention to both salient visual information and the user query at each decoding step. This reduces the impact of visual attention sink, as irrelevant tokens exhibit minimal shifts, while ensuring balanced cross-modal fusion for well-integrated representation. Extensive experiments show that GIFT effectively mitigates hallucination in VLMs across both generative and classification tasks, achieving up to 20.7\% improvement over greedy decoding, while maintaining general vision-language performance with low computational overhead.

## Setup

1. Create and activate conda environment:
```bash
conda create -n gift python=3.12
conda activate gift
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Install modified transformers:
```bash
cd transformers-4.50.0
pip install -e .
cd ..
```

## Supported Models

The following Vision-Language Models are currently supported:

- LLaVA-1.5-7B
- LLaVA-1.5-13B  
- Qwen2-VL-7B


## Configuration

Configuration files in the `configs/` directory specify:
- Model parameters
- Data paths
- GIFT parameters
  - `use_gift`: Enable/disable GIFT enhancement
  - `visual_saliency_computation_layers`: Layers used for computing visual saliency maps
  - `attention_enhancement_layers`: Layers where cross-modal attention is enhanced
  - `alpha`: Scaling factor for vision attention enhancement
  

Example configuration (llava_1.5_7b.yaml):
```yaml
model_name: "llava_1.5_7b"
max_new_tokens: 1
use_gift: true
visual_saliency_computation_layers: [11]
attention_enhancement_layers: [12,13,14,15,16,17,18,19,20,21,22]
alpha: 5.0
```

Default hyperparameters are available in the provided config files.

## Usage

Run inference with default settings:

```bash
python inference.py --config configs/llava_1.5_7b.yaml
```

## Modified Transformers Files
The implementation modifies the following Transformers files:
- transformers-4.50.0/src/transformers/generation/utils.py
- transformers-4.50.0/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
- transformers-4.50.0/src/transformers/models/llama/modeling_llama.py


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the CC BY-NC 4.0 License.

