# Fused MoE Training for Qwen3-MoE Models

This directory contains example configurations for training Qwen3-MoE models with fused MoE kernels, providing ~5x speedup over the default HuggingFace implementation.

## How It Works

The `custom_model_class` parameter allows LLaMA-Factory to load arbitrary model classes instead of using HuggingFace's Auto* classes. This enables using pre-converted models with optimized architectures.

## Prerequisites

1. Install the fused kernel package:
```bash
pip install git+https://github.com/woct0rdho/transformers-qwen3-moe-fused.git
```

2. Convert your model to fused format (one-time operation):
```python
from qwen3_moe_fused.convert import convert_model_to_fused
convert_model_to_fused("Qwen/Qwen3-30B-A3B", "/path/to/Qwen3-30B-A3B-fused")
```

## Usage

1. Update the config with your fused model path:
```yaml
model_name_or_path: /path/to/Qwen3-30B-A3B-fused
```

2. Run training:
```bash
llamafactory-cli train examples/extras/fused_moe/qwen3_moe_fused_sft.yaml
```

## Post-Training: Convert Back for Inference

After training, convert your model back to standard HuggingFace format for compatibility with vLLM, SGLang, and other inference engines:

```python
from qwen3_moe_fused.convert import convert_model_to_unfused
convert_model_to_unfused("/path/to/trained-model", "/path/to/trained-model-unfused")
```

The unfused model can then be used with any standard inference engine.

## Important Notes

- **ZeRO-2 Required**: MoE models are incompatible with ZeRO-3. Always use ZeRO-2.
- **pure_bf16**: Recommended to avoid fp32 upcasting which doubles memory usage.
- **Lossless Conversion**: The fused/unfused conversion is lossless - it's just reshaping tensors.

## Supported Models

- Qwen3-30B-A3B
- Other Qwen3-MoE variants

## Workflow Summary

```
Qwen/Qwen3-30B-A3B (HF format)
        │
        ▼ convert_model_to_fused()
        │
Qwen3-30B-A3B-fused (optimized format)
        │
        ▼ Train with LLaMA-Factory (5x faster)
        │
trained-model-fused
        │
        ▼ convert_model_to_unfused()
        │
trained-model (HF format) → vLLM/SGLang inference
```
