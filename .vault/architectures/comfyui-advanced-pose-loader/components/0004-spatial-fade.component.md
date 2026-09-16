---
type: component
c4_level: component
title: "Spatial Fade Component"
system: comfyui-advanced-pose-loader
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [fade, gradient, spatial, component]
see_also: ["concepts/0006-spatial-fade.concept.md"]
---

# Component: Spatial Fade

## Diagram

```mermaid
C4Component
    Container(advanced_loader, "Advanced OpenPose Loader", "Custom Node")
    Component(mask_generator, "Mask Generator", "_create_spatial_fade_mask()")
    Component(fade_applier, "Fade Applier", "_apply_spatial_fade()")
    Component(context_builder, "Context Builder", "_build_control_context()")
    
    Rel(mask_generator, fade_applier, "Provides fade mask")
    Rel(fade_applier, context_builder, "Applies fade to mask channels")
```

## Elements

| ID | Name | Type | Description |
|----|------|------|-------------|
| advanced_loader | Advanced OpenPose Loader | Container | Parent custom node |
| mask_generator | Mask Generator | Component | `_create_spatial_fade_mask()` — generates linear gradient mask (top/bottom/left/right) |
| fade_applier | Fade Applier | Component | `_apply_spatial_fade()` — applies fade mask to control context |
| context_builder | Context Builder | Component | `_build_control_context()` — receives masked context |

## Notes

Encapsulates spatial fade mask generation and application. Computes linear gradients on the 4 mask channels of the 260-dim control context.
