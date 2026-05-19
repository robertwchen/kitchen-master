# Time-To-Decision Benchmark

Measured for the three demo events using the existing `run_foot_fault_pipeline` event-decision stage with active side treated as known/overridden to isolate the foot-decision stage. These timings are offline compute measurements on this machine, not a deployed streaming benchmark.

## Compute Time

| Metric | Value |
|---|---:|
| Median 3-event decision run | 1.5773 s |
| Mean compute per event across 3-event runs | 533.78 ms/event |
| Mean isolated compute per event | 556.35 ms/event |

## Per-Event Isolated Compute

| Event frame | Timestamp (s) | Active side | Mean compute ms | Median compute s | Labels observed |
|---:|---:|---|---:|---:|---|
| 929 | 15.4981 | right override | 538.36 | 0.5384 | `["foot_fault_volley", "foot_fault_volley", "foot_fault_volley"]` |
| 1537 | 25.641 | right override | 612.11 | 0.6057 | `["uncertain", "uncertain", "uncertain"]` |
| 1948 | 32.4975 | left override | 518.58 | 0.5232 | `["legal_volley", "legal_volley", "legal_volley"]` |

## Algorithmic Delay From Temporal Context

| Stage | Future frames needed | Delay at 59.943 fps | Meaning |
|---|---:|---:|---|
| foot_localization_temporal_smoothing | 1 | 16.68 ms | needs neighboring frames around event to smooth foot contact point |
| bounce_confirmation | 5 | 83.41 ms | needs future ball trajectory to confirm bounce-like reversal |
| active_side_context | 0 | 0.0 ms | uses ball context near event; exact causal delay depends on whether implemented one-sided or centered in deployment |
| hit_classification_lookback | 0 | 0.0 ms | uses past bounce history before a hit; no future delay if hit frame is already known |

## Report-Safe Interpretation

After a hit/event frame is known, the configured foot decision uses a ±1-frame temporal window, which implies about 16.68 ms of future-frame algorithmic delay at 59.943 fps. The measured event-decision compute stage averaged about 556.35 ms per isolated event on this machine. If the event itself must be inferred from ball trajectory, the bounce logic can add up to 83.41 ms of future-frame delay from `lookahead_frames`. Because active side was overridden in this benchmark, these labels should not be reported as autonomous demo accuracy. End-to-end live deployment would still need a streaming benchmark that includes ball detection, event inference, foot localization, rendering/output, and p95/p99 latency.
