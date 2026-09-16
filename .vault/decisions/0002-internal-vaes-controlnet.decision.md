---
type: decision
id: DEC-0002
title: "Internal VAE and ControlNet Management"
status: accepted
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [architecture, vae, controlnet, caching]
see_also: ["concepts/0003-vaes-and-latent-space.concept.md"]
---

# DEC-0002: Internal VAE and ControlNet Management

## Context

Users would normally need to wire explicit VAE encoding nodes and ControlNet loading nodes into their workflows. This increases complexity and error potential.

## Decision

The node internally manages VAE loading, caching, and pose image encoding (`flux2-vae.safetensors`), and loads the FLUX.2 Fun ControlNet model (`FLUX.2-dev-Fun-Controlnet-Union-2602-fp8.safetensors`) with CPU placement to avoid VRAM conflicts. VAE is cached via `_vae_cache` for subsequent calls.

## Alternatives Considered

- Require user to wire VAE nodes explicitly — increases workflow complexity.
- Load ControlNet to GPU — causes VRAM conflicts with base model.

## Consequences

- Simplifies user workflow to a single node with pose folder name input.
- CPU loading prevents VRAM conflicts but may impact performance on low-VRAM systems.
- Caching improves performance across multiple runs.
