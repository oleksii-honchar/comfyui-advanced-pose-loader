---
type: container
c4_level: container
title: "Advanced OpenPose Loader Custom Node"
system: comfyui-advanced-pose-loader
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [comfyui, node, container]
see_also: ["decisions/0001-dual-mechanism-approach.decision.md"]
---

# Container: Advanced OpenPose Loader Custom Node

## Diagram

```mermaid
C4Container
    Person(user, "ComfyUI User", "Creates pose assets and runs workflows")
    System(comfyui, "ComfyUI", "Web-based image generation interface")
    
    Container(advanced_loader, "Advanced OpenPose Loader", "Python Custom Node", "Single-file custom node consolidating pose-based image generation pipeline")
    ContainerDb(pose_assets, "Pose Assets", "PNG Images", "Folder-per-pose structure with six pose types each")
    ContainerDb(flux_models, "FLUX.2 Models", "GGUF/Safetensors", "Base model, VAE, ControlNet, CLIP, LoRAs")
    
    Rel(user, comfyui, "Creates workflows and pose assets")
    Rel(comfyui, advanced_loader, "Invokes during sampling")
    Rel(advanced_loader, pose_assets, "Loads pose images")
    Rel(advanced_loader, flux_models, "Loads VAE and ControlNet models")
```

## Elements

| ID | Name | Type | Technology | Description |
|----|------|------|------------|-------------|
| user | ComfyUI User | Person | — | Creates pose assets and runs workflows |
| comfyui | ComfyUI | System | — | Web-based image generation interface |
| advanced_loader | Advanced OpenPose Loader | Container | Python Custom Node | Single-file custom node consolidating pose-based image generation pipeline |
| pose_assets | Pose Assets | ContainerDb | PNG Images | Folder-per-pose structure with six pose types each |
| flux_models | FLUX.2 Models | ContainerDb | GGUF/Safetensors | Base model, VAE, ControlNet, CLIP, LoRAs |

## Notes

The custom node itself is a container within ComfyUI. It manages its own internal state (VAE cache, ControlNet model) and exposes a simple interface to the ComfyUI workflow system.
