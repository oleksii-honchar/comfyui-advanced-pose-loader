---
type: concept
title: "VAEs and Latent Space"
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [vae, latent, encoding, decoding]
see_also: ["decisions/0002-internal-vaes-controlnet.decision.md", "concepts/0002-control-context-architecture.concept.md"]
---

# Concept: VAEs and Latent Space

## What

A VAE (variational auto-encoder) compresses pixel-space images into a smaller latent space. The encoder shrinks an 1024×1024 RGB image to a 64×64×(latent channels) tensor; the decoder reverses it.

## Why

During diffusion, the model denoises in latent space, not pixel space. Pose guide images must be converted to latents so the ControlNet context is expressed in the same latent space the denoiser works in.

## Key Details

- `flux2-vae.safetensors` is the VAE used (built for FLUX.2 family)
- 16× downsampling: 1024×1024 → 64×64×16 tensor
- FLUX.2 latent format uses 16 channels with quantized latents
- Sharp latent rounding is applied by the Flux2 latent format
- VAE is internally loaded once and cached across runs (`_vae_cache`)
- Decoder used only at end (VAEDecode → SaveImage)
