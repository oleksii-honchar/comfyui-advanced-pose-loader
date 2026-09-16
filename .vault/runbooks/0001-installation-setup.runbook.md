---
type: runbook
title: "Installation and Setup"
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [installation, setup, dependencies]
see_also: ["specifications/0001-pose-asset-structure.spec.md"]
---

# Runbook: Installation and Setup

## Prerequisites

- ComfyUI installed and running
- `comfyui-flux2fun-controlnet` custom node installed
- Sufficient VRAM (recommended: 12 GB+)

## Steps

1. Place `advanced_openpose_loader.py` in your ComfyUI `custom_nodes/` directory
2. Download the FLUX.2-dev-Fun-Controlnet-Union model:
   - Source: https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union
   - File: `FLUX.2-dev-Fun-Controlnet-Union-2602-fp8.safetensors`
   - Location: ComfyUI `controlnet/` directory
3. Download the FLUX.2 VAE:
   - File: `flux2-vae.safetensors` (or `ae.safetensors`)
   - Location: ComfyUI `vae/` directory
4. Download the FLUX.2-Klein base model (e.g., `Miraclein_4.3_9b_q8_0.gguf`):
   - Location: ComfyUI `checkpoints/` directory
5. Download CLIP text encoder (e.g., `flux2-klein-qwen3-q8_0.gguf`):
   - Location: ComfyUI `clip/` directory
6. Create pose assets folder structure (see pose asset structure spec)
7. Restart ComfyUI

## Verification

- Node appears in Custom/Pose category in ComfyUI node search
- No import errors in ComfyUI console
