---
type: component
c4_level: component
title: "VAE Encoder Component"
system: comfyui-advanced-pose-loader
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [vae, encoding, component]
see_also: ["concepts/0003-vaes-and-latent-space.concept.md"]
---

# Component: VAE Encoder

## Diagram

```mermaid
C4Component
    Container(advanced_loader, "Advanced OpenPose Loader", "Custom Node")
    Component(vae_loader, "VAE Loader", "_load_vae()")
    Component(vae_cache, "VAE Cache", "_vae_cache")
    Component(pose_encoder, "Pose Encoder", "_encode_pose_image()")
    Component(resizer, "Pose Resizer", "_resize_pose_image()")
    
    Rel(vae_loader, vae_cache, "Caches loaded VAE")
    Rel(pose_encoder, vae_cache, "Uses cached VAE")
    Rel(resizer, pose_encoder, "Prepares image for encoding")
```

## Elements

| ID | Name | Type | Description |
|----|------|------|-------------|
| advanced_loader | Advanced OpenPose Loader | Container | Parent custom node |
| vae_loader | VAE Loader | Component | `_load_vae()` method — loads flux2-vae.safetensors |
| vae_cache | VAE Cache | Component | `_vae_cache` — internal cache for loaded VAE |
| pose_encoder | Pose Encoder | Component | `_encode_pose_image()` — encodes pose images to latent space |
| resizer | Pose Resizer | Component | `_resize_pose_image()` — resizes to 1024×1024 |

## Notes

Encapsulates all VAE-related operations. The VAE is loaded once, cached, and reused across runs. Pose images are resized to 1024×1024 before encoding.
