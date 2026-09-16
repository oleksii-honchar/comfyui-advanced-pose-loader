---
type: memory
title: "Internal VAE Caching Behavior"
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [vae, caching, performance]
see_also: ["decisions/0002-internal-vaes-controlnet.decision.md", "concepts/0003-vaes-and-latent-space.concept.md"]
---

# Memory: Internal VAE Caching Behavior

## Fact

The node loads the VAE once and caches it in `_vae_cache` for reuse across subsequent runs within the same ComfyUI session.

## Context

This means the VAE is not reloaded for each pose image or each sampling run — only loaded the first time the node is used.

## Impact

- Performance: faster subsequent runs
- VRAM: VAE stays in memory after first use
- If you change VAE models between runs, you must restart ComfyUI or clear the cache
- User VAE wiring is optional (internal loading is default)
