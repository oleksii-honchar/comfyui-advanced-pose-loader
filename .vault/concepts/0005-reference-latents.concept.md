---
type: concept
title: "FLUX.2-Klein Reference Latents"
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [reference, latents, klein, conditioning]
see_also: ["decisions/0001-dual-mechanism-approach.decision.md"]
---

# Concept: FLUX.2-Klein Reference Latents

## What

A secondary conditioning pathway where pose latents are stored in conditioning metadata (`reference_latents` / `reference_latents_method`) for use by the FLUX.2-Klein base model.

## Why

Supports the refcontrol LoRA family (e.g., refcontrol_v2_poses_flux2_klein.safetensors) that rely on reference-latent conditioning for pose guidance.

## Key Details

- Optional pathway — ControlNet alone drives primary conditioning
- Patched `Flux.forward_orig` excludes reference-latent tokens from controlnet hints
- Main image tokens only receive the summed controlnet hints
- Enables pose-specific fine-tuning via LoRAs
