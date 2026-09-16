---
type: container
c4_level: lifecycle
title: "Flux Patch Lifecycle — Advanced OpenPose Loader"
system: comfyui-advanced-pose-loader
createdAt: "2026-09-16T09:00:00Z"
updatedAt: "2026-09-16T09:00:00Z"
tags: [patch, lifecycle, lifecycle, monkey-patch]
see_also: ["memories/0001-flux-patch-lifecycle.memory.md"]
---

# Flux Patch Lifecycle

## Diagram

```mermaid
stateDiagram-v2
    [*] --> Unpatched
    Unpatched --> Patching: ControlNetWrapper.pre_run()
    Patching --> VersionGuard: Inspect Flux.forward_orig signature
    VersionGuard --> Patched: Signatures match, apply patch
    VersionGuard --> PatchFailed: Core drift detected
    PatchFailed --> [*]
    
    Patched --> Sampling: Sampling begins
    Sampling --> GenerateHints: Each sampling step
    GenerateHints --> RunControlNet: _generate_flux2fun_hints()
    RunControlNet --> ApplyHints: _apply_flux2fun_hints()
    ApplyHints --> Sampling: Continue sampling
    
    Sampling --> CleaningUp: Sampling complete
    CleaningUp --> Unpatched: ControlNetWrapper.cleanup() → remove_patch()
    Unpatched --> [*]
    
    note right of VersionGuard
        Version-guards via inspect module
        Refuses to patch on core drift
    end note
    
    note right of GenerateHints
        Control context moved to
        img.device/dtype
        Hints generated per step
    end note
    
    note right of ApplyHints
        Hints added onto main
        image tokens only
        Summed across controlnets
    end note
```

## Notes

State diagram showing the complete lifecycle of the scoped monkey-patch applied to Flux.forward_orig. Highlights the version-guard mechanism that ensures compatibility with ComfyUI core updates.
