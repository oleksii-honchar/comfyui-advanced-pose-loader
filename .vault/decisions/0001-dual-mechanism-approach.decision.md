---
type: decision
id: DEC-0001
title: "Dual-Mechanism Pose Conditioning Approach"
status: accepted
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [architecture, pose, conditioning]
see_also: ["concepts/0004-flux-controlnet-union.concept.md", "concepts/0005-reference-latents.concept.md"]
---

# DEC-0001: Dual-Mechanism Pose Conditioning Approach

## Context

Pose-conditioned image generation requires multiple conditioning signals. ComfyUI has no built-in node that consumes a folder of pose assets and fuses six independent control signals into one positive/negative conditioning pair.

## Decision

Use a dual-mechanism approach:

1. **FLUX.2 Fun ControlNet (primary)** — Uses Alibaba's FLUX.2-dev-Fun-Controlnet-Union model. Provides primary pose guidance through latent-space conditioning via ControlNetWrapper pattern.
2. **FLUX.2-Klein Reference Latents (secondary)** — Stores pose latents in conditioning metadata for use by the FLUX.2-Klein base model through `reference_latents` and `reference_latents_method` metadata keys.

## Alternatives Considered

- Single ControlNet approach only — would miss the reference-latent pathway needed for refcontrol LoRAs.
- Pixel-space ControlNet (sd-style) — ruled out; ControlNet operates entirely on latent-space context.

## Consequences

- Node consolidates what would normally be a complex ControlNet subgraph into a single node.
- Users get professional pose-guided image generation without building multi-node subgraphs.
- Higher user strength = tighter pose lock (due to strength inversion semantics).
