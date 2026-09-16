---
type: runbook
title: "Usage Workflow"
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [usage, workflow, sampling]
see_also: ["decisions/0003-strength-inversion.decision.md"]
---

# Runbook: Usage Workflow

## Prerequisites

- Node installed and ComfyUI running
- Pose assets prepared (see pose asset structure spec)

## Steps

1. Load the FLUX.2-Klein base model using `UnetLoaderGGUF` node
2. Load the CLIP text encoder using `CLIPLoaderGGUF` node
3. (Optional) Load pose-specific LoRA (e.g., `refcontrol_v2_poses_flux2_klein.safetensors`)
4. Connect the model and conditioning to the Advanced OpenPose Loader node
5. Specify the pose folder name in the `folder_name` input
6. Adjust ControlNet strengths as needed (remember: higher = tighter lock)
7. (Optional) Configure spatial fade for gradient-based guidance
8. Connect outputs to your KSampler node
9. Configure KSampler:
   - Flux2/Klein (distilled): 4 steps, CFG 1.0, Flux Guidance node at ~3.5
   - Flux2 dev (non-distilled): Flux2Scheduler, CFG 0–1, Flux Guidance node
10. Generate

## Verification

- Output image shows pose-guided generation
- Pose matches the input pose assets
- Strength settings produce desired level of pose adherence
