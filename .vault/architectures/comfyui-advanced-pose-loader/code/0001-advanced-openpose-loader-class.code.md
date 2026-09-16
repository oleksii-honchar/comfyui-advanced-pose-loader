---
type: code
c4_level: code
title: "AdvancedOpenposeLoader Class"
system: comfyui-advanced-pose-loader
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [class, code, main]
see_also: ["components/0001-vae-encoder.component.md", "components/0002-controlnet-wrapping.component.md", "components/0003-pose-asset-handler.component.md", "components/0004-spatial-fade.component.md"]
---

# Code: AdvancedOpenposeLoader Class

## Diagram

```mermaid
C4Code
    Class_Boundary(node, "AdvancedOpenposeLoader Node") {
        Class(loader, "AdvancedOpenposeLoader", "Main node class") {
            Method(advanced_openpose_load, "(model, vae, controlnet, folder_name, conditioning, ...) -> (model, positive, negative)")
            Method(_get_poses_base_path, "() -> path")
            Method(_load_vae, "() -> vae")
            Method(_resize_pose_image, "(image) -> resized_image")
            Method(_encode_pose_image, "(path) -> latent")
            Method(_load_pose_image, "(folder_name, pose_type) -> image")
            Method(_store_reference_latents, "(conditioning, pose_latent)")
            Method(_get_controlnet_patch_functions, "(model_version) -> (patch, unpatch)")
            Method(_load_controlnet, "(model_path) -> controlnet")
            Method(_build_control_context, "(pose_latent) -> context")
            Method(_create_spatial_fade_mask, "(mode, strength) -> mask")
            Method(_apply_spatial_fade, "(context, mask) -> faded_context")
            Method(_apply_controlnet, "(model, conditioning, controlnet, strength)")
        }
        Class_Boundary(config, "Configuration") {
            Field(STRENGTH_DEFAULTS, "Dict of default strengths per pose type")
            Field(_vae_cache, "Cached VAE instance")
        }
    }
    
    Rel(loader, config, "Uses configuration")
```

## Elements

| ID | Name | Type | Description |
|----|------|------|-------------|
| loader | AdvancedOpenposeLoader | Class | Main node class with all pipeline methods |
| advanced_openpose_load | Method | Method | Main entry point — orchestrates entire pipeline |
| _get_poses_base_path | Method | Method | Resolves pose assets base path |
| _load_vae | Method | Method | Loads VAE model internally |
| _resize_pose_image | Method | Method | Resizes pose image to 1024×1024 |
| _encode_pose_image | Method | Method | Encodes pose image to latent space |
| _load_pose_image | Method | Method | Loads specific pose image from folder |
| _store_reference_latents | Method | Method | Stores pose latent in conditioning metadata |
| _get_controlnet_patch_functions | Method | Method | Gets patch functions for model version |
| _load_controlnet | Method | Method | Loads Flux2 Fun ControlNet model |
| _build_control_context | Method | Method | Builds 260-dim control context |
| _create_spatial_fade_mask | Method | Method | Creates spatial fade mask |
| _apply_spatial_fade | Method | Method | Applies spatial fade to context |
| _apply_controlnet | Method | Method | Applies ControlNet wrapper to conditioning |

## Notes

Single-file custom node class with 13 methods covering the entire pose-conditioned image generation pipeline. Category: Custom/Pose.
