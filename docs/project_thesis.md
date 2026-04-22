# Project Thesis

KitchenMaster investigates whether a portable side-profile camera placed near a pickleball net can reliably classify a player's foot position relative to the kitchen (non-volley zone, NVZ) line during volley-relevant moments, enabling objective analysis of potential kitchen faults in a constrained visual setup.

## Why NVZ Detection Matters

Accurate NVZ boundary interpretation is central to rule enforcement, player feedback, and fair play analysis in pickleball. A practical line-relative foot-state detector could support officiating studies, coaching workflows, and post-hoc review, especially in settings where full-court camera infrastructure is unavailable.

## Version 1 Scope

Version 1 is limited to classifying foot state relative to the kitchen line:
- `behind_line`
- `on_line`
- `over_line`
- `uncertain`

The focus is definition clarity, experiment scaffolding, and baseline evaluation mechanics.

## Not Included in Version 1

- Full game-event understanding (e.g., volley context detection)
- Player tracking across long sequences
- Multi-camera fusion
- Real-time officiating guarantees
- Production-grade deployment or hardware integration
