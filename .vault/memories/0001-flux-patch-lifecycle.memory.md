---
type: memory
title: "Flux.forward_orig Scoped Monkey-Patch Lifecycle"
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [patch, lifecycle, flux, sampling]
see_also: ["concepts/0004-flux-controlnet-union.concept.md"]
---

# Memory: Flux.forward_orig Scoped Monkey-Patch Lifecycle

## Fact

The node uses a version-guarded scoped monkey-patch of `Flux.forward_orig` from the `comfyui-flux2fun-controlnet` plugin (`flux_patch.py`).

## Lifecycle

1. `ControlNetWrapper.pre_run()` → `apply_patch()` — applied at start of sampling run
2. Version-guards signature via `inspect`; refuses to patch on core drift
3. When patched but no active Flux2Fun controlnet, patched fn delegates verbatim to captured original
4. Each sampling step, `_generate_flux2fun_hints()` runs per active controlnet
5. `_apply_flux2fun_hints()` adds hints onto main image tokens only
6. `ControlNetWrapper.cleanup()` → `remove_patch()` — removed after sampling

## Context

This is why the patch is not process-wide or import-time. It's scoped to the sampling run and auto-restored.

## Impact

- Safe for ComfyUI core compatibility (version-guards)
- Multiple controlnets at same layer are summed
- Hints resized spatially (bilinear) if token count drifts
