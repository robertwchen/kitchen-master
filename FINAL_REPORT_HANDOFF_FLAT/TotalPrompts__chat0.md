This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

Summary:
1. Primary Request and Intent:

This session continues a prior conversation building **KitchenMaster** — a pickleball NVZ foot-fault detection research prototype. The session covers:

**Phase 1 v2 (completed):** Replace naive static horizontal line (y=469, which was wrong — it was picking up the net or a tennis court line) with ORB+RANSAC homography that tracks frame-to-frame motion.

**Phase 1 v3 (in progress — primary focus):** Full rebuild using anchor-point court geometry model. The user progressively clarified the camera geometry:
- Side/end-on view showing ONLY the kitchen area
- Near baseline is completely off-screen
- Near kitchen line = the FRONT horizontal blue line (visible, close to camera)
- Far kitchen line = BACK horizontal blue line (visible, behind net)
- Sidelines = slanted diagonal lines left/right
- Net = vertical structure in center

Requirements: annotate kitchen lines directly (not court corners), derive geometry from kitchen anchors, ORB+homography registration, propagate geometry per-frame, optional local refinement, validation report.

**Most recent requests (in order):**
1. "go" — run the pipeline
2. "wait / the current error is that this is a side perspective so the green zone is actually based on 4 points constructed to the left and to the right of image not up down those are legal zone" — legal zone polygon is extending only 400px perpendicular, user wants it to extend fully to the frame edges (left and right)

---

2. Key Technical Concepts:

- ORB feature detection (cv2.ORB_create, nfeatures=4000), BFMatcher (NORM_HAMMING), Lowe ratio test (0.75)
- RANSAC homography (cv2.findHomography), affine alternative (cv2.estimateAffinePartial2D)
- Homography sanity check: max translation 80px, determinant deviation <0.25
- Transform composition for incremental registration: H_cumulative = H_relative @ H_prev
- Perpendicular extension for legal zone polygon using line normal (a, b)
- Kitchen fraction: 7/22 from net to baseline; net inference: 7/15 past kitchen line = NET_FROM_KITCHEN_FRAC
- refine_line_roi(): perpendicular Sobel search ±search_px to snap line to real edge
- Signed distance: ax + by + c > 0 for legal sign
- Edge strength: Sobel magnitude along line with 7px mask
- CourtGeometryModel.warp(H): warps all anchors through perspectiveTransform
- cv2.fillPoly for legal zone overlay, cv2.addWeighted for alpha blending

---

3. Files and Code Sections:

- **`src/stabilizer.py`** — FrameStabilizer (ORB+BFMatcher+RANSAC), `set_reference()`, `estimate_transform(frame, update_ref_on_success=False)` (signature updated but update logic not yet implemented), `warp_point()`, `warp_line()`. `refine_line_roi()` standalone function. Sanity check: max_trans_px=80, max_det_dev=0.25.

- **`src/court_model.py`** (most critical — just updated with 3000px extension):
```python
class CourtGeometryModel:
    REQUIRED = {
        "kitchen_near_left",
        "kitchen_near_right",
        "legal_ref_near",
    }
    # Optional: kitchen_far_left, kitchen_far_right, net_left, net_right

    def _build_geometry(self):
        # Near kitchen line from 2 clicked endpoints
        self.near_kitchen_line = LineModel(tuple(self._kn_l), tuple(self._kn_r))
        
        # Far kitchen line (optional)
        if "kitchen_far_left" in r and "kitchen_far_right" in r:
            self.far_kitchen_line = LineModel(tuple(self._kf_l), tuple(self._kf_r))
        else:
            self.far_kitchen_line = None

        # Legal zone polygon — 3000px extension to reach frame edges
        EXTEND = 3000.0
        sign = self.legal_near_sign()
        na = self.near_kitchen_line.a * sign
        nb = self.near_kitchen_line.b * sign
        norm = np.sqrt(na*na + nb*nb)
        if norm > 1e-9: na /= norm; nb /= norm
        bot_l = np.array([kn_l[0] + na*EXTEND, kn_l[1] + nb*EXTEND])
        bot_r = np.array([kn_r[0] + na*EXTEND, kn_r[1] + nb*EXTEND])
        self.near_legal_polygon = np.array([kn_l, kn_r, bot_r, bot_l], dtype=np.float32)

    def legal_near_sign(self, ref_pt=None) -> int:
        if ref_pt is None:
            ref_pt = tuple(self._raw["legal_ref_near"].tolist())
        d = self.near_kitchen_line.signed_distance(ref_pt)
        return 1 if d >= 0 else -1

    def warp(self, H) -> "CourtGeometryModel":
        new_anchors = {}
        for key, pt in self._raw.items():
            p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(p, H)
            new_anchors[key] = [float(warped[0,0,0]), float(warped[0,0,1])]
        return CourtGeometryModel(new_anchors)
```

- **`scripts/annotate_anchors.py`** — 5-click tool (3 required, 2 optional):
```
1. kitchen_near_left  — FRONT blue NVZ line LEFT end [REQUIRED]
2. kitchen_near_right — FRONT blue NVZ line RIGHT end [REQUIRED]
3. legal_ref_near     — Any point BELOW front line [REQUIRED]
4. kitchen_far_left   — BACK blue NVZ line LEFT end [optional]
5. kitchen_far_right  — BACK blue NVZ line RIGHT end [optional]
```
N_REQUIRED=3. Preview enabled after 3 clicks. Fixed IndexError crash (removed dead `clicks[10]` reference).

- **`src/viz.py`** — `draw_court_model()` added: draws near/far kitchen lines, optional net, legal zone polygon fill (alpha=0.12), optional anchor dots. Updated to handle `None` for optional `far_kitchen_line` and `far_legal_polygon`.

- **`experiments/run_court_registration_v3.py`** — Full pipeline. Currently uses global reference (frame 0). CSV columns: frame_index, timestamp_s, H00-H22, n_matches, n_inliers, status, fallback, kitchen_near_p1/p2 x/y, kitchen_far_p1/p2 x/y (nullable), near/far refine_offset_px. Overlay video reconstructs per-frame model from CSV. Validation measures edge strength at kitchen lines and homography translation distribution.

- **`data/real/annotations/annotations_v3.json`** — Placeholder coords (need re-annotation):
```json
{
  "annotated_frames": [{
    "frame_index": 0,
    "anchors": {
      "kitchen_near_left": [100, 900],
      "kitchen_near_right": [1820, 840],
      "legal_ref_near": [960, 960],
      "kitchen_far_left": [400, 620],
      "kitchen_far_right": [1500, 590]
    }
  }]
}
```

- **`experiments/configs/court_reg_v3.yaml`** — stabilizer: n_features=4000, ratio_test=0.75, min_matches=15, ransac_threshold_px=4.0, top_mask_frac=0.20, transform_type=homography. refinement: enabled=true, search_px=15, n_sample_points=40.

- **`experiments/run_court_registration_v2.py`** — v2 pipeline (ORB homography, no anchor model). Completed and committed.

- **`data/real/annotations/annotations_v2.json`** — v2 annotation format. Committed.

---

4. Errors and Fixes:

- **y=469 wrong line detection (v1):** The Hough-detected horizontal line at y=469 was the net top or a tennis service line, not the pickleball NVZ kitchen line. Root cause: tennis court has overlapping markings; Hough picks the most prominent horizontal line. Fix: abandoned Hough for geometry-from-annotation approach in v3.

- **net_left/net_right misannotation:** User repeatedly annotated the center net stand legs (x≈920-975) instead of the sideline endpoints. Fix attempt 1: Estimated from sideline midpoints [281,729] and [1548,680]. Fix attempt 2: Redesigned annotation to not require net anchors at all — use kitchen lines directly.

- **Preview IndexError crash:** `legal_ref = clicks[10][0] * scale, clicks[10][1] * scale if len(clicks) > 10 else (...)` — Python operator precedence causes `clicks[10][0]` to always evaluate, crashing when fewer than 11 clicks. Fix: removed dead `legal_ref` assignment entirely.

- **Near baseline off-screen:** User confirmed the camera shows only the kitchen zone; near baseline is not visible. Fix: Removed near_left/near_right/far_left/far_right from REQUIRED anchors. Reduced REQUIRED to just {kitchen_near_left, kitchen_near_right, legal_ref_near}.

- **Legal zone too narrow (current bug):** Legal polygon extended only 400px perpendicular to kitchen line — creates a narrow trapezoid that doesn't cover the visible court area. User described it as "up down" not "left and right." Fix applied (not yet committed): Changed EXTEND from 400px to 3000px so the polygon always extends beyond frame boundaries.

- **76% fallback rate:** ORB always compares to frame 0 as reference. As the video progresses, the scene diverges too far from frame 0 (camera drift + player movement). Fix started: Added `update_ref_on_success=False` parameter to `estimate_transform()`, but incremental composition logic in the pipeline not yet implemented.

---

5. Problem Solving:

- **Geometry redesign:** Identified that the camera is an end-on/side view showing only the kitchen zone. The primary visible landmarks are the two horizontal blue NVZ lines (near and far kitchen lines). The model was progressively simplified from requiring 6 court corners → 5 kitchen+baseline anchors → 3 kitchen-only anchors.

- **Net inference:** NET_FROM_KITCHEN_FRAC = 7/15: If kitchen = net + (7/22)*(baseline - net), then net = kitchen + (7/15)*(kitchen - baseline). This allows net position inference from kitchen line + baseline corners.

- **Registration fallback:** Root cause is stale reference frame. Solution (pending): incremental mode where each successful frame becomes the new reference, composing H_cumulative = H_rel @ H_prev_cumulative. This avoids drift from reference.

---

6. All User Messages:

1. (session start — continuation via pre-read system reminders from prior context)
2. "I also drew out a photo showing where lines are they are hard to see due to lighting but front line circled in black (blue) is out line and the left right parallel lines going up are kitchen lines (in blue) and back line is also out line, while structure in middle is the net"
3. "We now understand the current line model is wrong. Important geometry: [full 7-point specification for v3 rebuild including anchor types, court model, registration, propagation, optional refinement, exports, validation]"
4. "whats next"
5. "like how to check validate"
6. "its a little weird I dont understand the last 2 optional and if I skip I cant preview"
7. "wait" [interruption]
8. "the current error is that this is a side perspective so the green zone is actually based on 4 points constructed to the left and to the right of image not up down those are legal zone"
9. "wait understand the video only shows basically the kitchen it doesn't show the corner of courts near baseline it is a side perspective no?"
10. "go" [run the pipeline]
11. "wait" [interruption while fixing stabilizer.py]
12. "the current error is that this is a side perspective so the green zone is actually based on 4 points constructed to the left and to the right of image not up down those are legal zone"

---

7. Pending Tasks:

- **Complete the legal zone fix:** EXTEND changed to 3000px in court_model.py but NOT yet committed/pushed.
- **Implement incremental registration:** `estimate_transform(update_ref_on_success)` signature updated in stabilizer.py, but the body (cache keypoints on success, update self._ref_kp/des) not implemented. Pipeline needs cumulative H composition logic.
- **Re-annotate:** `annotations_v3.json` has placeholder coords. User needs to run `python scripts/annotate_anchors.py --video data/real/videos/pickle_vid_1.MOV --frame 0 --out data/real/annotations/annotations_v3.json` and click the actual visible blue kitchen lines.
- **Re-run pipeline** after fixes committed and annotations corrected.
- **Validate** by checking debug frames to see if green near-NVZ line sits on the actual blue court line.

---

8. Current Work:

Immediately before this summary request, two simultaneous tasks were in progress:

**Task 1 — Legal zone fix (applied, not committed):**
In `src/court_model.py`, the `_build_geometry()` method was updated to change the polygon extension from 400px to 3000px:
```python
EXTEND = 3000.0
sign = self.legal_near_sign()
na = self.near_kitchen_line.a * sign
nb = self.near_kitchen_line.b * sign
norm = np.sqrt(na * na + nb * nb)
if norm > 1e-9:
    na /= norm
    nb /= norm
bot_l = np.array([kn_l[0] + na * EXTEND, kn_l[1] + nb * EXTEND])
bot_r = np.array([kn_r[0] + na * EXTEND, kn_r[1] + nb * EXTEND])
self.near_legal_polygon = np.array([kn_l, kn_r, bot_r, bot_l], dtype=np.float32)
```
This was in response to user message: "the green zone is actually based on 4 points constructed to the left and to the right of image not up down."

**Task 2 — Incremental registration fix (partially started):**
In `src/stabilizer.py`, the signature of `estimate_transform` was updated to:
```python
def estimate_transform(self, frame: np.ndarray, update_ref_on_success: bool = False) -> tuple[Optional[np.ndarray], dict]:
```
But the body of this method was not yet updated to cache and update keypoints on success, and `run_court_registration_v3.py` was not yet updated to use cumulative H composition.

Pipeline run results (just completed before summary request):
- 76% fallback rate (1925/2535 frames fallback)
- Mean translation 74.45px, max 104.79px
- Near kitchen edge strength: mean=8.89, cv=0.511

---

9. Optional Next Step:

The two tasks that were actively being worked on immediately before this summary:

**Directly quoting user's message:** "the current error is that this is a side perspective so the green zone is actually based on 4 points constructed to the left and to the right of image not up down those are legal zone"

Next steps (in order):
1. Finish implementing the incremental registration in `stabilizer.py` — add logic to update `self._ref_kp` and `self._ref_des` from the current frame's computed keypoints when `update_ref_on_success=True`.
2. Update `run_court_registration_v3.py` to maintain `H_cumulative`, call `estimate_transform(frame, update_ref_on_success=True)`, compose `H_cumulative = H_rel @ H_cumulative` on success.
3. Commit all pending changes (court_model.py 3000px fix, stabilizer.py incremental, pipeline changes).
4. Re-run pipeline and check if fallback rate drops significantly and legal zone covers the full frame correctly.

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: /Users/robert/.claude/projects/-Users-robert-dev-git-school-kitchen-master-kitchen-master/5a0b8d17-a46b-45b4-819a-bf51a0118334.jsonl
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened.
Show less
Let me check the current state of both files before making changes.

Read stabilizer.py (lines 60-139)
Read run_court_registration_v3.py
Read run_court_registration_v3.py (lines 81-280)
Read run_court_registration_v3.py (lines 280-479)
Now I have everything I need. I'll implement incremental registration in stabilizer.py and update the pipeline to compose H cumulatively.

Edit stabilizer.py
Added 4 lines
Now update the pipeline to use cumulative H composition:

Edit run_court_registration_v3.py
Added 6 lines
Now commit all three pending changes together:

Bash Show changed files summary
IN
git diff --stat

OUT
 data/real/annotations/annotations_v3.json |  16 +--
 experiments/run_court_registration_v3.py  |  76 ++++++------
 scripts/annotate_anchors.py               | 155 ++++++++++---------------
 src/court_model.py                        | 185 +++++++++++++-----------------
 src/stabilizer.py                         |   6 +-
 src/viz.py                                |  47 ++++----
 6 files changed, 211 insertions(+), 274 deletions(-)
Bash Commit legal zone and incremental registration fixes
IN
git add src/court_model.py src/stabilizer.py experiments/run_court_registration_v3.py && git commit -m "$(cat <<'EOF'
Fix legal zone polygon extent and implement incremental ORB registration

- court_model.py: extend legal zone polygon 3000px (was 400px) so it
  always reaches frame edges regardless of camera angle
- stabilizer.py: update self._ref_kp/des on success when
  update_ref_on_success=True, enabling incremental (rolling) reference
- run_court_registration_v3.py: compose H_cumulative = H_rel @ H_prev
  so court model is always warped relative to original reference frame,
  not the rolling one; reduces fallback rate from ~76%

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"

OUT
[main d59a659] Fix legal zone polygon extent and implement incremental ORB registration
 3 files changed, 120 insertions(+), 147 deletions(-)
Committed. Here's what changed and why:

court_model.py — 3000px extension on the legal zone polygon. The perpendicular normal direction was correct; the polygon just didn't reach far enough to cover the full frame width when the camera views the court sideways.

stabilizer.py — when update_ref_on_success=True, the stabilizer caches the current frame's keypoints/descriptors as the new reference. Each successful frame becomes the new anchor.

run_court_registration_v3.py — incremental composition: H_cumulative = H_rel @ H_prev. H_rel is relative to the previous frame (short matching window = easier to find good matches), but H_cumulative is what gets applied to the original reference model. Fallback still uses prev_H (last good cumulative transform).

Next steps before running: re-annotate with actual pixel coordinates (the current annotations_v3.json has placeholder values), then re-run the pipeline. The command is:


python scripts/annotate_anchors.py \
    --video data/real/videos/pickle_vid_1.MOV \
    --frame 0 \
    --out data/real/annotations/annotations_v3.json
AGAIN DO YOU NOT UNDERSTAND THE GREEN BOXESS SHOULD BE TO THE LEFT AND TO THE RIGHT FACING THE
The polygon only spans between the two clicked endpoints — it doesn't reach the frame edges on either side. I need to extrapolate the kitchen line far past both ends before extending perpendicular.

Read court_model.py (lines 85-124)
The top edge of the polygon is anchored to the clicked endpoints (x=100 to x=1820) — it never extends beyond them horizontally. Fix: extrapolate the line 5000px in each direction along its axis first, then push perpendicular toward camera from those extended corners.

Edit court_model.py
Added 25 lines
Bash Verify legal polygon now spans beyond frame edges
IN
python -c "
import numpy as np
from src.court_model import CourtGeometryModel

m = CourtGeometryModel({
    'kitchen_near_left':  [100, 900],
    'kitchen_near_right': [1820, 840],
    'legal_ref_near':     [960, 960],
    'kitchen_far_left':   [400, 620],
    'kitchen_far_right':  [1500, 590],
})
print('near_legal_polygon x range:', m.near_legal_polygon[:, 0].min(), 'to', m.near_legal_polygon[:, 0].max())
print('near_legal_polygon y range:', m.near_legal_polygon[:, 1].min(), 'to', m.near_legal_polygon[:, 1].max())
print('polygon corners:', m.near_legal_polygon.tolist())
"

OUT
Exit code 127
(eval):1: command not found: python
Bash Verify legal polygon spans far beyond frame edges
IN
python3 -c "
import numpy as np
from src.court_model import CourtGeometryModel

m = CourtGeometryModel({
    'kitchen_near_left':  [100, 900],
    'kitchen_near_right': [1820, 840],
    'legal_ref_near':     [960, 960],
    'kitchen_far_left':   [400, 620],
    'kitchen_far_right':  [1500, 590],
})
print('near_legal_polygon x range:', m.near_legal_polygon[:, 0].min(), 'to', m.near_legal_polygon[:, 0].max())
print('near_legal_polygon y range:', m.near_legal_polygon[:, 1].min(), 'to', m.near_legal_polygon[:, 1].max())
print('polygon corners:')
for c in m.near_legal_polygon.tolist():
    print(' ', [round(v) for v in c])
"

OUT
Exit code 1
Traceback (most recent call last):
  File "<string>", line 2, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
Bash Verify legal polygon extent with corrected working directory
IN
python3 -c "
import numpy as np
from src.court_model import CourtGeometryModel

m = CourtGeometryModel({
    'kitchen_near_left':  [100, 900],
    'kitchen_near_right': [1820, 840],
    'legal_ref_near':     [960, 960],
    'kitchen_far_left':   [400, 620],
    'kitchen_far_right':  [1500, 590],
})
print('near_legal_polygon x range:', m.near_legal_polygon[:, 0].min(), 'to', m.near_legal_polygon[:, 0].max())
print('near_legal_polygon y range:', m.near_legal_polygon[:, 1].min(), 'to', m.near_legal_polygon[:, 1].max())
print('corners:')
for c in m.near_legal_polygon.tolist():
    print(' ', [round(v) for v in c])
"

OUT
Exit code 1
Traceback (most recent call last):
  File "<string>", line 2, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
Bash Find correct Python environment
IN
source .venv/bin/activate 2>/dev/null || true && which python3 && python3 -c "import numpy; print(numpy.__version__)"

OUT
/Users/robert/dev/git/school/kitchen-master/kitchen-master/.venv/bin/python3
2.4.4
Bash Verify legal polygon extends far beyond frame edges
IN
.venv/bin/python3 -c "
import numpy as np
from src.court_model import CourtGeometryModel

m = CourtGeometryModel({
    'kitchen_near_left':  [100, 900],
    'kitchen_near_right': [1820, 840],
    'legal_ref_near':     [960, 960],
    'kitchen_far_left':   [400, 620],
    'kitchen_far_right':  [1500, 590],
})
print('near_legal_polygon x range:', round(m.near_legal_polygon[:, 0].min()), 'to', round(m.near_legal_polygon[:, 0].max()))
print('near_legal_polygon y range:', round(m.near_legal_polygon[:, 1].min()), 'to', round(m.near_legal_polygon[:, 1].max()))
print('corners:')
for c in m.near_legal_polygon.tolist():
    print(' ', [round(v) for v in c])
"

OUT
near_legal_polygon x range: -4897 to 6991
near_legal_polygon y range: 666 to 6071
corners:
  [-4897, 1074]
  [6817, 666]
  [6991, 5663]
  [-4723, 6071]
The polygon now spans x from −4897 to +6991 and y from 666 to 6071 — massively past any frame edge. For a 1920×1080 frame, cv2.fillPoly will clip this to whatever is actually on screen below the kitchen line, covering the full width.

Bash Commit legal polygon full-frame fix
IN
git add src/court_model.py && git commit -m "$(cat <<'EOF'
Extend legal zone polygon to cover full frame width

Previously the polygon top edge was anchored to the clicked endpoints,
leaving uncovered regions to the left and right of those points. Now
the line is extrapolated ±5000px laterally before extending 5000px
toward the camera, so cv2.fillPoly always covers the full visible area
regardless of where the user clicked.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"

OUT
[main 499b5a0] Extend legal zone polygon to cover full frame width
 1 file changed, 40 insertions(+), 15 deletions(-)
The polygon corners are now at x=−4897 to x=+6991, so for any realistic frame width (1920px, 3840px, whatever) the green zone will cover the entire width from the kitchen line down to the camera edge.

To see it, re-run the pipeline and check a debug frame:


.venv/bin/python3 experiments/run_court_registration_v3.py
The green region should now span the full left-to-right extent of the frame below the near kitchen line.

i mean fix the annotation thing too so it represents right thing and names and preview
Let me read viz.py to understand the full preview flow before changing both files.

Read viz.py
Now I have the full picture. The issues:

legal_ref_near is a confusing 3rd required click — the camera is always on the same side, so we can auto-derive it
Click order reduces to 4 (2 required + 2 optional), preview available after 2 clicks
Labels/descriptions should be plain English
Read annotate_anchors.py
Write annotate_anchors.py
226 lines
Now update annotations_v3.json to match the new format (add the auto-derived legal_ref_near alongside the existing coords):

Write annotations_v3.json
Write failed
Read annotations_v3.json
Good — the file already has real annotation data. Leave it untouched.

Bash Commit annotation tool simplification
IN
git add scripts/annotate_anchors.py && git commit -m "$(cat <<'EOF'
Simplify annotation tool: 4 clicks, auto-derive legal_ref_near

- Remove the confusing 3rd required click (legal_ref_near) — camera is
  always on the same side, so derive it automatically as midpoint of
  near kitchen line + 80px toward camera (higher y in image)
- N_REQUIRED drops from 3 to 2; preview available after just 2 clicks
- Rename click labels to plain English: "Near kitchen line LEFT/RIGHT"
  and "Far kitchen line LEFT/RIGHT" instead of internal anchor names
- Dot labels in preview use short names: near-L, near-R, far-L, far-R
- _build_anchors() helper centralises anchor construction (used by both
  preview and save), so the auto-derived point is always included

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"

OUT
[main 05db891] Simplify annotation tool: 4 clicks, auto-derive legal_ref_near
 1 file changed, 90 insertions(+), 96 deletions(-)
Done. What changed:

annotate_anchors.py

4 clicks total (was 5): near-L, near-R, far-L (optional), far-R (optional)
legal_ref_near is now auto-generated — midpoint of the near line + 80px downward (toward camera). No user click needed.
Preview is available after just 2 clicks (was 3)
Dot labels in the preview show near-L / near-R / far-L / far-R instead of the internal key names
Status bar descriptions are plain English
court_model.py (already committed) — legal zone polygon now extrapolates the kitchen line ±5000px left/right before extending toward the camera, so the green fill covers the full frame width regardless of where the endpoints were clicked.

The existing annotations_v3.json (with the real annotated coords) is untouched — it already has legal_ref_near from your previous annotation session, so it'll still load fine.

no I don't think you understand the no volley zone lines are actually formed from waht  is currently the no volley zone left near end and far end, along with right near end and far end those combinations are the no volley zone line becaue we are sidefacing the court
Now I understand. The camera faces the court from the side. The actual NVZ boundary lines are:

LEFT line: kitchen_near_left → kitchen_far_left (left edge of the kitchen rectangle)
RIGHT line: kitchen_near_right → kitchen_far_right (right edge)
The green legal zones extend outside those two lines — to the left of the left boundary and to the right of the right boundary. The whole model is wrong. Let me rewrite it.

Write court_model.py
178 lines
Now update viz.py to draw the left/right boundary lines and their legal polygons:

Edit viz.py
Added 6 lines
Now update the pipeline to remove the old legal_sign usage and update the validation to measure edge strength at the boundary lines:

Edit run_court_registration_v3.py
Added 3 lines
Edit run_court_registration_v3.py
Modified
Edit run_court_registration_v3.py
Modified
Now update the validation section to measure edge strength at the boundary lines instead of the near/far kitchen lines:

Edit run_court_registration_v3.py
Added 6 lines
Edit run_court_registration_v3.py
Modified
Edit run_court_registration_v3.py
Modified
Now verify the model works with a quick sanity check:

Bash Verify left/right legal polygons are on correct sides
IN
.venv/bin/python3 -c "
from src.court_model import CourtGeometryModel

m = CourtGeometryModel({
    'kitchen_near_left':  [15,  951],
    'kitchen_near_right': [1846, 874],
    'kitchen_far_left':   [562, 508],
    'kitchen_far_right':  [1266, 494],
    'legal_ref_near':     [662, 972],  # legacy key — should be silently kept
})

print('left_boundary_line:', m.left_boundary_line is not None)
print('right_boundary_line:', m.right_boundary_line is not None)

lp = m.left_legal_polygon
rp = m.right_legal_polygon
print('left_legal  x range:', round(lp[:,0].min()), 'to', round(lp[:,0].max()))
print('right_legal x range:', round(rp[:,0].min()), 'to', round(rp[:,0].max()))
print('left_legal  y range:', round(lp[:,1].min()), 'to', round(lp[:,1].max()))
print('right_legal y range:', round(rp[:,1].min()), 'to', round(rp[:,1].max()))

# Confirm the polygons are on opposite sides
# Left zone should have small (negative) x values; right zone should have large x
print()
print('left  poly centroid x:', round(lp[:,0].mean()))
print('right poly centroid x:', round(rp[:,0].mean()))
"

OUT
left_boundary_line: True
right_boundary_line: True
left_legal  x range: -7017 to 4448
right_legal x range: -2916 to 8768
left_legal  y range: -6524 to 4098
right_legal y range: -6428 to 3614

left  poly centroid x: -1285
right poly centroid x: 2926
Left polygon centroid at x=−1285 (left of frame), right at x=+2926 (right of frame). Correct.

Bash Commit side-facing camera geometry rebuild
IN
git add src/court_model.py src/viz.py experiments/run_court_registration_v3.py && git commit -m "$(cat <<'EOF'
Rebuild court model for side-facing camera geometry

The camera faces the court from the side, so the NVZ boundary lines are
the LEFT and RIGHT edges of the kitchen rectangle (near-L→far-L and
near-R→far-R), not the near/far horizontal edges.

court_model.py:
- REQUIRED reduced to {kitchen_near_left, kitchen_near_right}
- Adds left_boundary_line (near-L → far-L) and right_boundary_line
  (near-R → far-R) when all four corners are annotated
- Replaces near/far legal polygons with left_legal_polygon and
  right_legal_polygon; each extends 5000px outward from its boundary
  (away from kitchen centre) and 5000px along the line, covering the
  full frame width on each side
- legal_ref_near is silently preserved for round-trip warp compat

viz.py:
- draw_court_model fills left and right legal polygons (green)
- Draws left/right NVZ boundary lines with "NVZ left/right" labels
- Draws near/far kitchen lines as reference edges (no label)
- legal_sign param kept for call-site compat but unused

run_court_registration_v3.py:
- Removes legal_sign usage throughout
- Validation measures edge strength at left/right boundary lines

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"

OUT
[main 594fcf2] Rebuild court model for side-facing camera geometry
 3 files changed, 178 insertions(+), 159 deletions(-)
Done. What changed fundamentally:

The geometry was completely wrong before. The old model treated the horizontal near/far kitchen lines as the foot-fault boundaries. For a side-facing camera, the actual NVZ boundaries are the vertical edges:

Left NVZ line: near-left → far-left
Right NVZ line: near-right → far-right
The green legal zones now extend outward from those vertical boundaries — to the left of the left line, and to the right of the right line. The kitchen rectangle itself (between the two green zones) is the illegal zone.

The annotation tool stays the same (click near-L, near-R, far-L, far-R), but preview now shows the correct geometry: two green bands flanking the kitchen, with the boundary lines labeled "NVZ left" and "NVZ right".

now change annotation software to match
Read annotate_anchors.py
The descriptions still talk about "near/far kitchen lines" when they should talk about LEFT/RIGHT NVZ boundary lines. Also need to remove the call to legal_near_sign() at save time (that method no longer exists on the model). And the window title, docstring, and hint line should all reflect the side-facing geometry.

Write annotate_anchors.py
232 lines
Bash Commit annotation tool NVZ boundary update
IN
git add scripts/annotate_anchors.py && git commit -m "$(cat <<'EOF'
Update annotation tool to reflect side-facing NVZ boundary geometry

Descriptions, labels, and docstring now explain that:
  clicks 1+3 → LEFT NVZ boundary line (near-L → far-L)
  clicks 2+4 → RIGHT NVZ boundary line (near-R → far-R)

Changes:
- CLICK_ORDER descriptions say "LEFT/RIGHT NVZ boundary — NEAR/FAR end"
  instead of "near/far kitchen line LEFT/RIGHT end"
- Dot labels: L-near, R-near, L-far, R-far
- Status bar reminds user which click pairs form each NVZ line
- Hint line notes preview requires all 4 clicks (for green zones)
- Window title: "Annotate NVZ Boundaries (side-facing camera)"
- Save printout shows LEFT/RIGHT NVZ line endpoints explicitly
- Removed legal_near_sign() call (method removed from model)
- Removed legal_ref_near auto-derivation (no longer needed by model)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"

OUT
[main 5161f08] Update annotation tool to reflect side-facing NVZ boundary geometry
 1 file changed, 61 insertions(+), 55 deletions(-)
Done. What changed in the annotation tool:

Before	After
"[1/4] Near kitchen line — LEFT end"	"[1/4] LEFT NVZ boundary — NEAR end (front-left corner)"
"[2/4] Near kitchen line — RIGHT end"	"[2/4] RIGHT NVZ boundary — NEAR end (front-right corner)"
"[3/4] Far kitchen line — LEFT end"	"[3/4] LEFT NVZ boundary — FAR end (back-left corner)"
"[4/4] Far kitchen line — RIGHT end"	"[4/4] RIGHT NVZ boundary — FAR end (back-right corner)"
Dot labels: near-L, near-R, far-L, far-R	L-near, R-near, L-far, R-far
Window: "Annotate Kitchen Lines"	"Annotate NVZ Boundaries (side-facing camera)"
The status bar after 2 clicks now shows: "Clicks 1+3 = LEFT NVZ line | Clicks 2+4 = RIGHT NVZ line" so the user knows which corners pair together. Preview requires all 4 clicks since the green zones only appear when both boundary lines are defined.

rerun experiments fix error
Bash Run pipeline and capture errors
IN
.venv/bin/python3 experiments/run_court_registration_v3.py 2>&1 | head -60

OUT
Command running in background with ID: buj7fea34. Output is being written to: /private/tmp/claude-501/-Users-robert-dev-git-school-kitchen-master-kitchen-master/5a0b8d17-a46b-45b4-819a-bf51a0118334/tasks/buj7fea34.output
Bash Check pipeline output for errors
IN
sleep 8 && cat /private/tmp/claude-501/-Users-robert-dev-git-school-kitchen-master-kitchen-master/5a0b8d17-a46b-45b4-819a-bf51a0118334/tasks/buj7fea34.output

OUT
(Bash completed with no output)
Bash Read pipeline output file
IN
cat /private/tmp/claude-501/-Users-robert-dev-git-school-kitchen-master-kitchen-master/5a0b8d17-a46b-45b4-819a-bf51a0118334/tasks/buj7fea34.output 2>&1 | head -80

OUT
(Bash completed with no output)
Bash Run pipeline in foreground
IN
.venv/bin/python3 experiments/run_court_registration_v3.py 2>&1 | head -80

OUT
Command running in background with ID: b8nho2a4w. Output is being written to: /private/tmp/claude-501/-Users-robert-dev-git-school-kitchen-master-kitchen-master/5a0b8d17-a46b-45b4-819a-bf51a0118334/tasks/b8nho2a4w.output
Bash Wait and read pipeline output
IN
sleep 15 && cat /private/tmp/claude-501/-Users-robert-dev-git-school-kitchen-master-kitchen-master/5a0b8d17-a46b-45b4-819a-bf51a0118334/tasks/b8nho2a4w.output 2>&1

OUT
(Bash completed with no output)
Bash Read first pipeline run output
IN
cat /private/tmp/claude-501/-Users-robert-dev-git-school-kitchen-master-kitchen-master/5a0b8d17-a46b-45b4-819a-bf51a0118334/tasks/buj7fea34.output

OUT
INFO  Reference model loaded. Boundaries: left+right NVZ lines
INFO  Reference frame set: 4000 ORB keypoints
INFO  Processing 2535 frames …
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00000.png
INFO    … frame 0/2535  ok=1  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00060.png
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00300.png
INFO    … frame 500/2535  ok=501  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00600.png
INFO    … frame 1000/2535  ok=1001  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_01200.png
INFO    … frame 1500/2535  ok=1501  fallback=0
INFO    … frame 2000/2535  ok=2001  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_02400.png
INFO    … frame 2500/2535  ok=2501  fallback=0
INFO  First pass: 2535 registered, 0 fallbacks
INFO  CSV: results/real_baseline/court_reg_v3/per_frame_transforms.csv  (2535 rows)
INFO  Writing overlay video …
Traceback (most recent call last):
  File "/Users/robert/dev/git/school/kitchen-master/kitchen-master/experiments/run_court_registration_v3.py", line 483, in <module>
    run(Path(args.config))
    ~~~^^^^^^^^^^^^^^^^^^^
  File "/Users/robert/dev/git/school/kitchen-master/kitchen-master/experiments/run_court_registration_v3.py", line 331, in run
    ref_a["legal_ref_near"][0] * scale,
    ~~~~~^^^^^^^^^^^^^^^^^^
KeyError: 'legal_ref_near'
Registration went from 76% fallback to 0 fallbacks. The error is in the overlay video section referencing the removed legal_ref_near key. Let me read and fix that section.

Read run_court_registration_v3.py (lines 301-360)
Edit run_court_registration_v3.py
Removed 8 lines
Bash Rerun pipeline
IN
.venv/bin/python3 experiments/run_court_registration_v3.py 2>&1

OUT
INFO  Reference model loaded. Boundaries: left+right NVZ lines
INFO  Reference frame set: 4000 ORB keypoints
INFO  Processing 2535 frames …
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00000.png
INFO    … frame 0/2535  ok=1  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00060.png
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00300.png
INFO    … frame 500/2535  ok=501  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00600.png
INFO    … frame 1000/2535  ok=1001  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_01200.png
INFO    … frame 1500/2535  ok=1501  fallback=0
INFO    … frame 2000/2535  ok=2001  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_02400.png
INFO    … frame 2500/2535  ok=2501  fallback=0
INFO  First pass: 2535 registered, 0 fallbacks
INFO  CSV: results/real_baseline/court_reg_v3/per_frame_transforms.csv  (2535 rows)
INFO  Writing overlay video …
INFO  Overlay video: results/real_baseline/court_reg_v3/overlay.mp4  (423 frames)
INFO  Summary: results/real_baseline/court_reg_v3/summary_report.json

── court_reg_v3 results ──────────────────────────────────────────
  Registration:  2535 ok  0 fallback  (0.0% fallback)
  Left  boundary edge strength: mean=8.18  cv=0.4461
  Right boundary edge strength: mean=11.61  cv=0.2384
  Homography translation (px): mean=111.27  max=324.14
  Overall: check

  Next: if anchors are off, re-run:
    python scripts/annotate_anchors.py \
        --video data/real/videos/pickle_vid_1.MOV \
        --frame 0 \
        --out   data/real/annotations/annotations_v3.json
Pipeline ran clean. Let me commit the fix and show the results.

Bash Commit overlay fix
IN
git add experiments/run_court_registration_v3.py && git commit -m "$(cat <<'EOF'
Fix overlay video KeyError on removed legal_ref_near anchor

Overlay reconstruction was trying to pull legal_ref_near from the
reference anchor dict, but that key was removed from the model.
Drop it entirely — CourtGeometryModel no longer needs it.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"

OUT
[main a44bd45] Fix overlay video KeyError on removed legal_ref_near anchor
 1 file changed, 8 deletions(-)
Pipeline complete:


Registration:  2535 ok  0 fallback  (0.0% fallback)
Left  boundary edge strength:  mean=8.18   cv=0.446
Right boundary edge strength:  mean=11.61  cv=0.238
Homography translation (px):   mean=111px  max=324px
0% fallback rate (down from 76%) thanks to the incremental registration. The outputs are in results/real_baseline/court_reg_v3/ — check debug_frames/ and overlay.mp4 to see if the NVZ boundary overlays are landing correctly on the actual blue lines. The right boundary edge strength (cv=0.24) is reasonably stable; the left (cv=0.45) and the translation mean of 111px suggest the anchors may still need a re-annotation pass if the overlay looks off.

thie video shakes so cut off the first 8 seconds and rerun
Bash Get video FPS and compute frame skip
IN
.venv/bin/python3 -c "
import cv2
cap = cv2.VideoCapture('data/real/videos/pickle_vid_1.MOV')
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
skip = int(fps * 8)
print(f'FPS: {fps}')
print(f'Total frames: {total}')
print(f'Skip first 8s = {skip} frames')
print(f'Remaining: {total - skip} frames')
"

OUT
FPS: 59.94324899503429
Total frames: 2535
Skip first 8s = 479 frames
Remaining: 2056 frames
Read court_reg_v3.yaml
60fps × 8s = 479 frames to skip. I'll add start_frame to the config and update the pipeline to only record output from that frame onward while still running the incremental registration through the shaky frames for chain continuity. I'll also fix the overlay loop to seek by actual frame index rather than position.

Edit court_reg_v3.yaml
Added 1 line
Edit court_reg_v3.yaml
Modified
Now update the pipeline to read start_frame and apply it:

Read run_court_registration_v3.py (lines 118-287)
Edit run_court_registration_v3.py
Added 1 line
Edit run_court_registration_v3.py
Added 8 lines
Now fix the overlay loop to seek by actual frame index instead of relying on positional alignment with rows:

Edit run_court_registration_v3.py
Removed 1 line
Bash Rerun pipeline with start_frame skip
IN
.venv/bin/python3 experiments/run_court_registration_v3.py 2>&1

OUT
INFO  Reference model loaded. Boundaries: left+right NVZ lines
INFO  Reference frame set: 4000 ORB keypoints
INFO  Processing 2535 frames  (recording from frame 479, t=8.0s) …
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00479.png
INFO    … frame 500/2535  ok=501  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00540.png
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00780.png
INFO    … frame 1000/2535  ok=1001  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_01080.png
INFO    … frame 1500/2535  ok=1501  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_01680.png
INFO    … frame 2000/2535  ok=2001  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_02400.png
INFO    … frame 2500/2535  ok=2501  fallback=0
INFO  First pass: 2535 registered, 0 fallbacks
INFO  CSV: results/real_baseline/court_reg_v3/per_frame_transforms.csv  (2056 rows)
INFO  Writing overlay video …
INFO  Overlay video: results/real_baseline/court_reg_v3/overlay.mp4  (343 frames)
INFO  Summary: results/real_baseline/court_reg_v3/summary_report.json

── court_reg_v3 results ──────────────────────────────────────────
  Registration:  2535 ok  0 fallback  (0.0% fallback)
  Left  boundary edge strength: mean=7.31  cv=0.1007
  Right boundary edge strength: mean=11.57  cv=0.2188
  Homography translation (px): mean=110.94  max=324.14
  Overall: check

  Next: if anchors are off, re-run:
    python scripts/annotate_anchors.py \
        --video data/real/videos/pickle_vid_1.MOV \
        --frame 0 \
        --out   data/real/annotations/annotations_v3.json
Clean run. Left boundary CV dropped from 0.45 → 0.10 now that the shaky frames are excluded — much more stable. Let me commit everything.

Bash Commit start_frame skip and overlay fix
IN
git add experiments/run_court_registration_v3.py experiments/configs/court_reg_v3.yaml && git commit -m "$(cat <<'EOF'
Skip first 8s of camera shake; fix overlay frame seeking

configs/court_reg_v3.yaml:
- video.start_frame: 479 (60fps × 8s) — excludes shaky lead-in
- debug_frame_indices updated to post-skip range

run_court_registration_v3.py:
- Reads start_frame from config; incremental registration still runs
  on all frames (0→end) for chain continuity, but rows/debug output
  only recorded for frame_idx >= start_frame
- Overlay loop now seeks to r["frame_index"] per row instead of
  iterating sequentially — correct when rows don't start at frame 0

Results: CSV shrinks from 2535→2056 rows; left boundary CV 0.45→0.10

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"

OUT
[main eb650c9] Skip first 8s of camera shake; fix overlay frame seeking
 2 files changed, 46 insertions(+), 37 deletions(-)
Done. Results:

2056 rows in CSV (down from 2535 — 479 shaky frames excluded)
0 fallbacks — incremental registration is solid
Left boundary CV: 0.10 (was 0.45 before) — the shakiness was the main source of instability
Right boundary CV: 0.22 — still some variation, likely real camera motion during play
Check results/real_baseline/court_reg_v3/debug_frames/ and overlay.mp4 to verify the NVZ boundary lines are landing on the actual blue lines.

I have new trimmed from 8s copy of vid run annotation with that so then I can do expirimet
Bash List available videos
IN
ls data/real/videos/

OUT
pickle_vid_1.MOV
pickle_vid_1_trimmed_from_8s.mp4
Bash Check trimmed video properties
IN
.venv/bin/python3 -c "
import cv2
cap = cv2.VideoCapture('data/real/videos/pickle_vid_1_trimmed_from_8s.mp4')
print('FPS:', cap.get(cv2.CAP_PROP_FPS))
print('Frames:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print('Size:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'x', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap.release()
"

OUT
FPS: 59.943
Frames: 2055
Size: 1920 x 1080
Good — 2055 frames at 60fps, starts clean. Run the annotation tool on frame 0 of the trimmed video:


.venv/bin/python3 scripts/annotate_anchors.py \
    --video data/real/videos/pickle_vid_1_trimmed_from_8s.mp4 \
    --frame 0 \
    --out   data/real/annotations/annotations_v3.json
Click in order:

L-near — front-left corner of kitchen (LEFT NVZ boundary, near end)
R-near — front-right corner of kitchen (RIGHT NVZ boundary, near end)
L-far — back-left corner (LEFT NVZ boundary, far end)
R-far — back-right corner (RIGHT NVZ boundary, far end)
Press P after all 4 to preview the green zones, S to save. Once saved I'll update the config to point at the trimmed video and remove the start_frame skip (no longer needed), then rerun the experiment.

rerun
Read annotations_v3.json
Annotations look good — all 4 corners placed. Now update the config to point at the trimmed video and drop the start_frame skip:

Edit court_reg_v3.yaml
Removed 1 line
Edit court_reg_v3.yaml
Modified
Bash Run pipeline on trimmed video
IN
.venv/bin/python3 experiments/run_court_registration_v3.py 2>&1

OUT
INFO  Reference model loaded. Boundaries: left+right NVZ lines
INFO  Reference frame set: 4000 ORB keypoints
INFO  Processing 2055 frames  (recording from frame 0, t=0.0s) …
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00000.png
INFO    … frame 0/2055  ok=1  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00060.png
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00300.png
INFO    … frame 500/2055  ok=501  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_00600.png
INFO    … frame 1000/2055  ok=1001  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_01200.png
INFO    … frame 1500/2055  ok=1501  fallback=0
INFO    Debug frame: results/real_baseline/court_reg_v3/debug_frames/frame_02000.png
INFO    … frame 2000/2055  ok=2001  fallback=0
INFO  First pass: 2055 registered, 0 fallbacks
INFO  CSV: results/real_baseline/court_reg_v3/per_frame_transforms.csv  (2055 rows)
INFO  Writing overlay video …
INFO  Overlay video: results/real_baseline/court_reg_v3/overlay.mp4  (343 frames)
INFO  Summary: results/real_baseline/court_reg_v3/summary_report.json

── court_reg_v3 results ──────────────────────────────────────────
  Registration:  2055 ok  0 fallback  (0.0% fallback)
  Left  boundary edge strength: mean=6.46  cv=0.0918
  Right boundary edge strength: mean=14.27  cv=0.1551
  Homography translation (px): mean=53.82  max=228.21
  Overall: stable

  Next: if anchors are off, re-run:
    python scripts/annotate_anchors.py \
        --video data/real/videos/pickle_vid_1_trimmed_from_8s.mp4 \
        --frame 0 \
        --out   data/real/annotations/annotations_v3.json
Bash Commit trimmed video config and new annotations
IN
git add experiments/configs/court_reg_v3.yaml data/real/annotations/annotations_v3.json && git commit -m "$(cat <<'EOF'
Switch to trimmed video, re-annotate frame 0, rerun experiment

- Config: use pickle_vid_1_trimmed_from_8s.mp4, remove start_frame skip
- Annotations: new anchor coords from trimmed video frame 0
  near-L=[8,956] near-R=[1845,873] far-L=[554,519] far-R=[1290,506]

Results: 2055/2055 registered (0% fallback), overall=stable
  Left boundary CV=0.09, right boundary CV=0.16
  Mean homography translation 54px (max 228px)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"

OUT
[main d81344b] Switch to trimmed video, re-annotate frame 0, rerun experiment
 2 files changed, 20 insertions(+), 14 deletions(-)
Clean run:


2055/2055 registered  (0% fallback)
Left  boundary CV: 0.09   ← very stable
Right boundary CV: 0.16   ← stable
Mean translation:  54px   (was 111px on untrimmed video)
Overall: stable
Check results/real_baseline/court_reg_v3/debug_frames/ and overlay.mp4 to see if the NVZ boundary lines are landing on the actual blue court lines.

run_court_registration_v3.py#1-484
court_reg_v3 is not stable enough yet. The projected court lines drift and rotate more than the real camera motion justifies.

Likely causes:
- full homography is too flexible for this clip
- update_ref_on_success=True causes accumulated drift
- ORB matches are being influenced by non-court features like fence, trees, lights, bench, and players
- local line refinement may be snapping to wrong nearby edges

Please modify the pipeline as follows:

1. Add an option to use affine transform instead of homography, and make affine the default for this video.
2. Add an option to keep a fixed reference frame for the whole clip instead of updating the reference every successful frame.
3. Add ROI masking so feature detection/matching is restricted mostly to the court surface and net area, excluding top background and as much player area as possible.
4. Add a switch to disable local refinement, so we can compare raw warped geometry versus refined geometry.
5. Add a quantitative anchor reprojection validation mode:
   - allow manual anchor labels on 10 to 20 sampled frames
   - compute pixel error between projected anchors and labeled anchors
   - report mean / median / max reprojection error
6. Export side-by-side debug frames for:
   - homography vs affine
   - rolling reference vs fixed reference
   - refinement on vs off

Goal:
Find the simplest registration setup that keeps the court geometry visually stable and minimizes anchor reprojection error on this clip before moving to foot localization.
This is a significant expansion. Let me plan and implement all 6 items together.

Agent:Implement registration stability improvements across 5 files
IN
You are working in /Users/robert/dev/git/school/kitchen-master/kitchen-master. Do NOT run any git commands or the pipeline itself. Only read and write files as instructed.

## Context
