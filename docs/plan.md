# Project Notes

## Completed

- Synthetic benchmark with 200 generated samples across legal, fault, borderline, and occluded cases.
- Baseline signed-distance classifier with `legal`, `fault`, and `uncertain` outputs.
- Anchor-based court geometry model for side-view NVZ boundaries.
- ORB and translation-based frame registration for real video.
- Ball tracking, volley candidate logic, foot localization, and foot-fault review pipeline.
- README-ready figures and small result summaries from the final report.
- Unit tests for generator, detector, evaluation, court geometry, and event-labeling code.

## Best Current Evidence

| Area | Evidence |
| --- | --- |
| Court registration | `2055 / 2055` frames registered with `0` fallbacks on the real demo clip. |
| Synthetic classifier | No false faults or missed faults on clear synthetic cases. |
| Human review workflow | Three saved real candidate events exported with annotated frames and event CSV rows. |

## Next Engineering Steps

- Collect a larger labeled real-event set across multiple courts and camera angles.
- Train or fine-tune a pickleball-specific ball detector.
- Improve active-side inference or add a lightweight operator-side selection step.
- Convert pixel-distance thresholds into calibrated court coordinates.
- Expand tests around registration CSV loading and full review-file override handling.
