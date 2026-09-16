---
type: container
c4_level: sequence
title: "Sampling Sequence — Advanced OpenPose Loader"
system: comfyui-advanced-pose-loader
createdAt: "2026-09-16T09:00:00Z"
updatedAt: "2026-09-16T09:00:00Z"
tags: [sequence, sampling, flow]
see_also: ["code/0001-advanced-openpose-loader-class.code.md"]
---

# Sampling Sequence: Advanced OpenPose Loader

## Diagram

```mermaid
sequenceDiagram
    participant ComfyUI as ComfyUI Workflow
    participant Loader as AdvancedOpenposeLoader
    participant PoseLoader as _load_pose_image()
    participant VAELoader as _load_vae()
    participant Encoder as _encode_pose_image()
    participant CNLoader as _load_controlnet()
    participant ContextBuilder as _build_control_context()
    participant Wrapper as _apply_controlnet()
    participant Patch as flux_patch.py
    participant Flux as Flux Model
    participant Sampler as KSampler
    
    ComfyUI->>Loader: advanced_openpose_load(model, vae, controlnet, folder_name, conditioning, ...)
    
    Loader->>PoseLoader: Load pose images from poses/{folder_name}/
    PoseLoader-->>Loader: Six pose images (1024x1024)
    
    Loader->>VAELoader: _load_vae()
    VAELoader-->>Loader: Loaded VAE (cached)
    
    Loader->>Encoder: _encode_pose_image() for each pose
    Encoder->>VAELoader: vae.encode(image)
    VAELoader-->>Encoder: Latent tensor [1,16,64,64]
    Encoder-->>Loader: Six pose latents
    
    Loader->>CNLoader: _load_controlnet()
    CNLoader-->>Loader: Loaded FLUX.2 Fun ControlNet model
    
    Loader->>ContextBuilder: _build_control_context(pose_latent)
    ContextBuilder-->>Loader: 260-dim control context [B,seq,260]
    
    Loader->>Wrapper: _apply_controlnet(model, conditioning, controlnet, strength)
    Wrapper->>Patch: ControlNetWrapper.pre_run() → apply_patch()
    Patch->>Flux: Monkey-patch Flux.forward_orig (version-guarded)
    Wrapper-->>Loader: Modified conditioning with ControlNetWrapper
    
    Loader-->>ComfyUI: (model, positive, negative)
    
    ComfyUI->>Sampler: KSampler(model, positive, negative, latent, seed, steps, cfg)
    
    loop Each sampling step (4 steps for Klein)
        Sampler->>Flux: forward(orig) with timestep, img, txt, guidance
        Flux->>Patch: Patched forward_orig called
        Patch->>ContextBuilder: _generate_flux2fun_hints()
        ContextBuilder->>CNLoader: Run controlnet on pose context
        CNLoader-->>ContextBuilder: Control hints per block
        Patch->>ContextBuilder: _apply_flux2fun_hints()
        ContextBuilder->>Flux: Add hints onto main image tokens only
        Flux->>Sampler: Denoised latent step
    end
    
    Sampler->>Patch: ControlNetWrapper.cleanup() → remove_patch()
    Patch->>Flux: Restore original Flux.forward_orig
    Sampler-->>ComfyUI: Final denoised latent
    
    ComfyUI->>VAELoader: VAEDecode(final_latent)
    VAELoader-->>ComfyUI: Generated image
```

## Notes

Complete sequence diagram showing the pose-conditioned image generation flow from workflow invocation through sampling to final image output. Highlights the scoped monkey-patch lifecycle and the dual-mechanism approach.
