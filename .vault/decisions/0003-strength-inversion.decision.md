---
type: decision
id: DEC-0003
title: "ControlNet Strength Inversion Semantics"
status: accepted
createdAt: "2026-09-16T08:55:00Z"
updatedAt: "2026-09-16T08:55:00Z"
tags: [strength, controlnet, semantics]
see_also: ["concepts/0007-pose-types-strengths.concept.md"]
---

# DEC-0003: ControlNet Strength Inversion Semantics

## Context

The plugin wrapper (`comfyui-flux2fun-controlnet`) scales hints by `1.0 - user_strength`. This creates counterintuitive semantics where higher user strength means tighter pose lock.

## Decision

Adopt the strength inversion rule: user-facing strengths are inverted when passed to the ControlNetWrapper (`1.0 - user_strength`). Document this clearly for users.

- User 0.75 → wrapper 0.25 (tight pose lock)
- User 0.0 → wrapper 1.0 (disabled)
- User 2.0 → wrapper -1.0 (over-override)

## Alternatives Considered

- Bug the wrapper authors to change semantics — would break compatibility.
- Document as-is — adopted.

## Consequences

- Users must understand the inversion rule or will invert their expectations.
- Symmetric across all six controlnets and matches the plugin wrapper's documented semantics.
