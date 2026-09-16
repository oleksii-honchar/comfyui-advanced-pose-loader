---
type: concept
title: "FLUX.2-dev-Fun-Controlnet-Union Model"
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [controlnet, flux, union, alibaba]
see_also: ["concepts/0002-control-context-architecture.concept.md", "decisions/0001-dual-mechanism-approach.decision.md"]
---

# Concept: FLUX.2-dev-Fun-Controlnet-Union Model

## What

Alibaba's unified ControlNet from the VideoX-Fun project. A single checkpoint supporting multiple control modes (canny, pose, depth, normal, color, mask, backdrop, blur, …).

## Why

The union architecture auto-detects the control type at runtime by reading whether the control channels are present/which channels are populated. No per-type model is needed — the same checkpoint handles all six pose types.

## Key Details

- Version: FLUX.2-dev-Fun-Controlnet-Union-2602-fp8.safetensors (fp8 build)
- Native VideoX-Fun architecture: Flux double-stream transformer augmented with control blocks
- Added on 4 double blocks
- Operates in latent space via control context token stream
- Loaded to CPU by default to avoid VRAM conflicts
