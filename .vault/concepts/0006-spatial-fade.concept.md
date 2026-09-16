---
type: concept
title: "Spatial Fade"
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [fade, gradient, mask, spatial]
see_also: ["concepts/0002-control-context-architecture.concept.md"]
---

# Concept: Spatial Fade

## What

Gradient-based pose guidance that varies pose influence across the image. Creates a linear gradient mask computed on the mask channels of the control context.

## Why

Useful for smooth transitions where pose guidance should be stronger in some areas than others (e.g., strong body guidance but weaker at edges for blending).

## Key Details

- Modes: `none`, `top`, `bottom`, `left`, `right`
- Strength: 0.0–1.0 (0.0 = no fade, 1.0 = maximum fade)
- Default strength: 0.5
- Implemented via `_create_spatial_fade_mask` method
- Modulates the 4 mask channels in the 260-dim control context
