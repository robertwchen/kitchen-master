# Photos For Slides

This folder contains the highest-value slide assets copied out of `results/` and renamed for easy use.

## Recommended Main Slides

- `01_registration_overlay.png`
  - Best single image for explaining the registered court geometry.

- `02_registration_comparison.png`
  - Side-by-side registration comparison for a method/ablation slide.

- `03_detected_fault_event.png`
  - Strongest end-to-end result image showing a detected foot fault.

- `04_uncertain_event.png`
  - Best image for explaining why the system uses an `uncertain` label.

## Optional Backup Slide

- `05_uncertain_review_event.png`
  - Another ambiguous event frame useful for Q&A or a limitations slide.

## Supporting Files

- `registration_overlay_video.mp4`
  - Short video artifact showing registration over time.

- `court_reg_v3_summary_report.json`
  - Contains key registration numbers such as `2055/2055` successful frames and `0` fallbacks.

- `demo_foot_fault_summary.json`
  - Demo snapshot summary: `3` events total, `1` fault, `2` uncertain.

- `demo_foot_fault_events.csv`
  - Per-event table with signed distances, foot points, and labels.

- `TECHNICAL_SUMMARY.md`
  - Long-form repo and results summary for reference while building slides.
