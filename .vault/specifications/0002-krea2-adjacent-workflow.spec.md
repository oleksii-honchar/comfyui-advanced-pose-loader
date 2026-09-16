---
type: specification
kind: guidance
title: "Krea2 Adjacent Workflow Guidance"
status: proposed
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [krea2, workflow, guidance]
see_also: ["memories/0002-krea2-is-not-flux2-klein.memory.md"]
---

# Specification: Krea2 Adjacent Workflow Guidance

## Goal

Provide workflow guidance for users who want to use pose conditioning with Krea2 models (separate Krea AI family), acknowledging the node itself is FLUX.2-only.

## Context

Krea2 is a separate Krea AI model family (RAW + Turbo, Krea AI Community License). The node's code has no Krea2 code paths. However, the same pose folder structure can be used with a Krea2 controlnet/union adapter in a separate workflow.

## Workflow

- **Krea2 RAW:** 52 steps, cfg 2.5, full quality (also used for LoRA training)
- **Krea2 Turbo:** 8 steps, cfg ~1.0, distilled for speed
- Use same pose folder structure: `poses/{pose_id}/{type}.png`
- Requires a Krea2 controlnet adapter (not included in this node)

## Risks

- ⚠️ Unexamined adapter path — if user's setup genuinely runs Krea2 through this loader, implies an unexamined adapter
- Node itself only tested with FLUX.2/FLUX.2-Klein
