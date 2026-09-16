---
type: index
title: "ComfyUI Advanced OpenPose Loader"
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: []
---

# ComfyUI Advanced OpenPose Loader

Single-file custom node that consolidates the entire pose-based image generation pipeline into one node.

## System Context

- [[0001-system-context.container]] — Complete system context with dependencies

## Flows and Sequences

- [[0002-sampling-sequence.container]] — Complete sampling sequence with patch lifecycle
- [[0003-control-context-flow.container]] — Control context data flow (260-dim context)
- [[0004-vae-encoding-flow.container]] — VAE encoding with caching
- [[0005-patch-lifecycle.container]] — Flux.forward_orig patch lifecycle state diagram
- [[0006-pose-asset-loading.container]] — Pose asset loading from folder structure

## C4 Levels

- [[containers/0001-advanced-openpose-loader.container]] — The custom node as a ComfyUI container
- [[components/0001-vae-encoder.component]] — Internal VAE loading and encoding
- [[components/0002-controlnet-wrapping.component]] — ControlNet loading and context construction
- [[components/0003-pose-asset-handler.component]] — Pose asset loading from folder structure
- [[components/0004-spatial-fade.component]] — Spatial fade mask generation
- [[code/0001-advanced-openpose-loader-class.code]] — Main node class and methods
