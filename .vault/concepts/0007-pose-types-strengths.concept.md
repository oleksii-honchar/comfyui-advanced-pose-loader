---
type: concept
title: "Pose Types and Strength Semantics"
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [pose, types, strength, defaults]
see_also: ["decisions/0003-strength-inversion.decision.md"]
---

# Concept: Pose Types and Strength Semantics

## What

The node supports six pose types, each with independent strength control and default values.

## Pose Types

| Type | Default Strength | Purpose |
|------|------------------|---------|
| `openpose` | 0.75 | Body pose from keypoints |
| `openpose_hand` | 0.80 | Hand pose detail |
| `openpose_full` | 0.85 | Full body including face |
| `canny` | 0.85 | Edge detection (Canny edges) |
| `depth` | 0.60 | Depth map guidance |
| `normal` | 0.40 | Normal vector guidance |

## Strength Semantics

- Range: 0.0–2.0 (0.0 = disabled)
- Inverted at wrapper: `wrapper_strength = 1.0 - user_strength`
- Higher user strength = tighter pose lock = weaker residual freedom

## File Naming

Each pose type loads a separate PNG: `poses/{pose_id}/{type}.png`
