# Project: KitchenMaster

## Goal

Build a first-pass portable side-view vision prototype for pickleball kitchen/NVZ foot-fault detection.

## Core Detection Task

Given a fixed side-view camera near the kitchen line, determine whether a player's foot:
1. stays legal behind the line,
2. touches/crosses the line and commits a fault, or
3. cannot be determined reliably and should be marked uncertain.

## Research Questions

- **RQ1**: Can a fixed side-view portable camera detect NVZ line contact in controlled conditions?
- **RQ2**: How sensitive is detection to viewpoint, blur, occlusion, and line-foot distance?
- **RQ3**: Can an uncertain output reduce wrong calls in ambiguous cases?

## Initial Scope

- One camera only
- One court region only (NVZ/kitchen line)
- Side-view profile only
- Controlled clips first
- Synthetic data first, then small real dataset

## Scenario Classes

| Scenario | Description |
|----------|-------------|
| **Clear legal** | The foot remains completely behind the kitchen line during the volley event window. |
| **Clear fault** | The foot clearly touches or crosses the kitchen line during the volley event window. |
| **Borderline contact** | The foot comes within a tiny margin of the line or appears to barely touch it, making the decision sensitive to resolution, blur, or calibration. |
| **Occluded / uncertain** | The foot-line relationship cannot be determined reliably because of occlusion, motion blur, bad lighting, or partial visibility. |

## Output Labels

- `legal`
- `fault`
- `uncertain`

## Evaluation Metrics

- Confusion matrix
- Precision per class
- Recall per class
- Uncertain rate
- False fault rate (legal called as fault)
- Missed fault rate (fault called as legal)

## Success Criteria for 3-Day Version

- Synthetic dataset generated successfully and saved
- Baseline detector runs end-to-end on synthetic data
- Small real dataset collected and labeled
- Preliminary metrics and failure cases documented honestly
