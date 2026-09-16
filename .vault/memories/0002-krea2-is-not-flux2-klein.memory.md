---
type: memory
title: "Krea2 is Not FLUX.2-Klein"
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [krea2, klein, flux, distinction]
see_also: ["specifications/0002-krea2-adjacent-workflow.spec.md"]
---

# Memory: Krea2 is Not FLUX.2-Klein

## Fact

Krea2 is a separate Krea AI model family (RAW + Turbo, Krea AI Community License), NOT FLUX.2-Klein. The node's code has no Krea2 code paths.

## Context

"Krea2" appeared in the user's original request scope alongside "flux2". Investigation confirmed they are distinct model families with different architectures, licensing, and sampling parameters.

## Impact

- Documentation should clarify this distinction
- Users wanting pose conditioning with Krea2 need a different controlnet adapter
- Node is FLUX.2/FLUX.2-Klein-only
