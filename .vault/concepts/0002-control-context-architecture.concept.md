---
type: concept
title: "Control Context Architecture"
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [control, context, latent, architecture]
see_also: ["concepts/0004-flux-controlnet-union.concept.md", "concepts/0003-vaes-and-latent-space.concept.md"]
---

# Concept: Control Context Architecture

## What

The node builds a 260-dimensional control context per spatial position, flattened to [batch, sequence_length, 260] format.

## Layout

- Channels 0–127: pose latent features (VAE-encoded pose image content)
- Channels 128–131: mask (4 channels, spatial fade)
- Channels 132–259: inpaint (128 channels, zeros; reserved)

## Why

This matches the VideoX-Fun "control context" convention expected by FLUX.2-dev-Fun-Controlnet-Union, a 4-block double-stream ControlNet trained on the Flux.2-dev base.

## Key Details

- The 260-dim vectors are laid out per spatial position matching the image latent layout
- ControlNet consumes this token-streamwise in the same layout as FLUX.2 image latents
- Every encoding step exists solely to produce this exact tensor shape
