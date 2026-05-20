# Problem

Pickleball NVZ, or kitchen, foot faults are hard to call from normal play. A legal volley depends on both the foot position and the timing of contact with the ball. From a single side-view camera, both signals are noisy.

KitchenMaster tests whether a fixed consumer camera can still produce useful review artifacts:

- registered NVZ boundary lines,
- candidate volley frames,
- estimated foot contact points,
- signed distance from the foot to the NVZ boundary,
- and an explicit `uncertain` output when the evidence is weak.

The project is framed as offline review, not autonomous officiating. Clear events can be flagged. Borderline or low-confidence events should be sent to a person.

## Labels

| Label | Meaning |
| --- | --- |
| `legal_volley` | Foot is clearly outside the NVZ boundary at the volley frame. |
| `foot_fault_volley` | Foot is clearly on or inside the NVZ boundary at the volley frame. |
| `uncertain` | Foot, ball timing, or active side is not reliable enough for a call. |

## Main Risks

- The ball is small and easy to lose in night footage.
- Foot localization can lock onto shadows, paddle motion, or the wrong lower-body blob.
- Active-side inference can select the wrong player side.
- Pixel thresholds depend on camera placement and resolution.
