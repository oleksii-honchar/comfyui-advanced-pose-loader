---
type: component
c4_level: component
title: "ControlNet Wrapping Component"
system: comfyui-advanced-pose-loader
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [controlnet, wrapping, component]
see_also: ["concepts/0002-control-context-architecture.concept.md", "concepts/0004-flux-controlnet-union.concept.md"]
---

# Component: ControlNet Wrapping

## Diagram

```mermaid
C4Component
    Container(advanced_loader, "Advanced OpenPose Loader", "Custom Node")
    Component(cn_loader, "ControlNet Loader", "_load_controlnet()")
    Component(context_builder, "Context Builder", "_build_control_context()")
    Component(wrapper_applicator, "Wrapper Applicator", "_apply_controlnet()")
    System_Ext(flux2fun, "comfyui-flux2fun-controlnet", "Plugin providing ControlNetWrapper")
    
    Rel(cn_loader, context_builder, "Provides loaded ControlNet model")
    Rel(context_builder, wrapper_applicator, "Provides 260-dim control context")
    Rel(wrapper_applicator, flux2fun, "Uses ControlNetWrapper for patching")
```

## Elements

| ID | Name | Type | Description |
|----|------|------|-------------|
| advanced_loader | Advanced OpenPose Loader | Container | Parent custom node |
| cn_loader | ControlNet Loader | Component | `_load_controlnet()` — loads FLUX.2 Fun ControlNet model |
| context_builder | Context Builder | Component | `_build_control_context()` — builds 260-dim control context |
| wrapper_applicator | Wrapper Applicator | Component | `_apply_controlnet()` — applies ControlNetWrapper with strength inversion |
| flux2fun | comfyui-flux2fun-controlnet | System_Ext | External plugin providing ControlNetWrapper |

## Notes

Encapsulates ControlNet loading, context construction, and wrapper application. The strength inversion (1.0 - user_strength) happens here. The plugin's scoped monkey-patch lifecycle is triggered via the wrapper.
