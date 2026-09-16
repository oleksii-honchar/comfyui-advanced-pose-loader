---
type: container
c4_level: system_context
title: "System Context — Advanced OpenPose Loader"
system: comfyui-advanced-pose-loader
createdAt: "2026-09-16T09:00:00Z"
updatedAt: "2026-09-16T09:00:00Z"
tags: [system, context, c4]
see_also: ["containers/0001-advanced-openpose-loader.container.md"]
---

# System Context: Advanced OpenPose Loader

## Diagram

```mermaid
C4Context
    Person(user, "ComfyUI User", "Creates pose assets, builds workflows, generates images")
    
    System_Boundary(comfyui_env, "ComfyUI Environment") {
        System(comfyui, "ComfyUI", "Web-based image generation interface")
        System(advanced_loader, "Advanced OpenPose Loader", "Custom node for pose-conditioned generation")
        System_Boundary(model_files, "Model Assets") {
            SystemDb(base_model, "FLUX.2-Klein Base Model", "GGUF checkpoint")
            SystemDb(controlnet_model, "FLUX.2 Fun ControlNet Union", "Safetensors model")
            SystemDb(vae_model, "FLUX.2 VAE", "Safetensors encoder/decoder")
            SystemDb(clip_model, "CLIP Text Encoder", "GGUF text encoding model")
            SystemDb(lora_models, "Pose LoRAs (optional)", "Refcontrol family fine-tuning")
        }
        SystemDb(pose_assets, "Pose Asset Folders", "PNG pose images (openpose, canny, depth, etc.)")
        SystemDb(output_images, "Generated Images", "Pose-conditioned output images")
    }
    
    Rel(user, comfyui, "Builds workflows and generates images")
    Rel(comfyui, advanced_loader, "Invokes node during sampling")
    Rel(advanced_loader, base_model, "Uses as base generation model")
    Rel(advanced_loader, controlnet_model, "Loads for pose conditioning")
    Rel(advanced_loader, vae_model, "Loads for image encoding/decoding")
    Rel(advanced_loader, clip_model, "Uses for text encoding")
    Rel(advanced_loader, lora_models, "Optionally loads for fine-tuning")
    Rel(advanced_loader, pose_assets, "Loads pose images from folders")
    Rel(comfyui, output_images, "Saves generated images")
    
    UpdateElementStyle(advanced_loader, $bgColor="#e6f3ff", $borderColor="#0066cc")
    UpdateElementStyle(pose_assets, $bgColor="#fff3e6", $borderColor="#cc6600")
```

## Elements

| ID | Name | Type | Description |
|----|------|------|-------------|
| user | ComfyUI User | Person | Creates pose assets, builds workflows, generates images |
| comfyui | ComfyUI | System | Web-based image generation interface |
| advanced_loader | Advanced OpenPose Loader | System | Custom node for pose-conditioned generation |
| base_model | FLUX.2-Klein Base Model | SystemDb | GGUF checkpoint (e.g., Miraclein 4.3 9B Q8) |
| controlnet_model | FLUX.2 Fun ControlNet Union | SystemDb | Safetensors model for pose conditioning |
| vae_model | FLUX.2 VAE | SystemDb | Safetensors encoder/decoder |
| clip_model | CLIP Text Encoder | SystemDb | GGUF text encoding model |
| lora_models | Pose LoRAs (optional) | SystemDb | Refcontrol family fine-tuning |
| pose_assets | Pose Asset Folders | SystemDb | PNG pose images (six types per pose) |
| output_images | Generated Images | SystemDb | Pose-conditioned output images |

## Notes

System context view showing the Advanced OpenPose Loader node's position within the broader ComfyUI environment and its dependencies.
