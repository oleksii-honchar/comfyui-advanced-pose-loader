---
type: concept
title: "Advanced OpenPose Loader Node"
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [node, overview, pose]
see_also: ["decisions/0001-dual-mechanism-approach.decision.md"]
---

# Concept: Advanced OpenPose Loader Node

## What

A single-file ComfyUI custom node that consolidates the entire pose-based image generation pipeline into one node. Takes a pose folder name as input and outputs modified model and positive/negative conditioning ready for KSampler.

## Why

Eliminates the need for a complex multi-node ControlNet subgraph. Makes pose-conditioned image generation accessible to users without deep knowledge of ControlNet wiring.

## Key Details

- Supports six pose types with independent strength controls
- Works with FLUX.2 and FLUX.2-Klein models
- Internally manages VAE, ControlNet, and context construction
- Production-ready with LoRA fine-tuning support
