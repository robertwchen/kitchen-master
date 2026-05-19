# KitchenMaster Final Report From Chat

> **Note:** This document is a **thorough reconstruction** of the work done across this chat, based on the conversation history and the project artifacts that were shared in the chat. It is **not a verbatim transcript** of every single message, but it is meant to capture the full technical story, the major questions asked, the design decisions, the code/files discussed, the outputs produced, the numbers used, and the presentation/report support that was built around the project.

---

# 1. Project Identity

**Project title:** KitchenMaster: Side-View Pickleball Foot-Fault Detection  
**Core idea:** Explore whether a **single fixed side-view camera** can detect pickleball kitchen / non-volley zone (NVZ) foot faults, and when the system should instead return **uncertain** rather than forcing a wrong call.

The project was repeatedly framed in chat as a **research prototype**, not a finished officiating product.

---

# 2. High-Level Research Framing Built in This Chat

Across this chat, the project converged on three central research questions:

1. **Can side-view footage detect foot contact relative to the kitchen line in controlled conditions?**
2. **How sensitive is the pipeline to blur, occlusion, and viewpoint changes?**
3. **Does returning uncertain reduce obviously wrong calls?**

This framing became important not only for the technical pipeline, but also for the final presentation and the eventual final report.

---

# 3. What I Asked About in This Chat

This section summarizes the major categories of questions and requests asked during the chat.

## 3.1 Dataset / feasibility / project direction
I asked about:
- finding a **large dataset** of pickleball side-profile footage with the camera aligned near the net and showing both kitchen lines
- whether existing footage online could be found instead of recording my own
- whether the side-profile setup was too difficult because the lines were hard to see
- what technology would have the **highest success rate** for this problem
- whether the model should first detect / reconstruct the court or box before reasoning about feet

## 3.2 Whether earlier work was still useful
I asked:
- what was the point of the earlier work if the line detection / geometry was still difficult on real video
- whether I should abandon the earlier simpler CV approaches and move to stronger models

## 3.3 What to do next in the repo
I asked:
- what the next step should be after extracting frames and wanting to “build something real”
- what prompt to give Cursor / Claude Code
- whether to use YAML-driven pipelines
- whether to keep different agents working on different parts of the system

## 3.4 How to validate the registration and line logic
I asked about:
- how to validate whether court registration was actually correct
- whether the orange line / moving line in overlays was wrong
- whether the net should be used as a reference
- what the current registration overlays meant
- why the tracker and smoothing did not look stable

## 3.5 Ball tracking / bounce / volley logic
I asked:
- whether bounce detection should come first before foot checking
- whether “ball goes down then back up” is enough to detect a bounce
- whether a more advanced segmentation model should be used for the ball
- what the best model/ranking was for ball bounce detection
- how to verify ball tracking step by step rather than blindly trusting it

## 3.6 Foot localization / foot-fault logic
I asked:
- how to get feet positioning correctly from event frames
- why not just use a body segmentation model and track feet
- whether to switch to a stronger pose / segmentation model
- how to interpret ambiguous event frames like frame 1948
- whether active-side inference was wrong in certain cases

## 3.7 Presentation building
I asked:
- how to structure a presentation for the project
- whether I should include the different approaches I tried, the failures, and the design pivots
- how to turn the story into slides
- how to make the slides less AI-looking and more like a normal student-built engineering research deck
- how to preview and download the slides

## 3.8 Final reporting
I asked for:
- a full final report in Markdown that captures everything from the chat, including the questions asked, what was done, code/files/numbers, and the overall story of the project

---

# 4. Core Technical Story That Emerged in the Chat

The project did **not** proceed as “train a model and solve it.” Instead, the chat established a layered research path:

1. **Synthetic baseline** to define the problem and evaluation logic.
2. **Real court registration** to estimate the correct court geometry from real side-view video.
3. **End-to-end demo pipeline** to connect court geometry, ball/event reasoning, foot localization, and final legal/fault/uncertain decisions.

A repeated theme in the chat was that the project’s strongest value came from showing:
- what failed,
- why it failed,
- what design decision changed because of that failure,
- and how the final pipeline became more principled and interpretable.

---

# 5. Early Existing Work That Was Already in the Repo / Chat Context

At one point in the chat, the existing repo status was summarized as already containing a substantial amount of foundational work. The user explicitly described that several tasks were already complete and live on GitHub at a specific commit, including:

- repo structure preserved
- `SampleMeta` dataclass
- baseline detector complete
- saved predictions / metrics / confusion matrix artifacts
- `failure_analysis.csv` grouped by scenario × occlusion × blur × distance bucket

That earlier synthetic work mattered because it gave the project:
- a defined label space,
- an interpretable first baseline,
- evaluation logic,
- and a way to talk about **uncertainty** as part of the system rather than an afterthought.

---

# 6. Phase 0: Synthetic Baseline

The chat repeatedly referred back to the synthetic baseline as the first phase of the project.

## 6.1 Purpose of the synthetic phase
The purpose was not to solve real pickleball footage immediately. It was to:
- define the task,
- create labels,
- build a first classical CV baseline,
- test signed-distance logic,
- establish evaluation metrics,
- and understand what kinds of errors mattered.

## 6.2 Synthetic outputs discussed
The synthetic phase was described as producing artifacts like:
- metadata CSV
- predictions CSV
- metrics CSV
- confusion matrix CSV / PNG
- qualitative overlays
- grouped failure analysis

## 6.3 Synthetic metrics referenced in chat
The strongest synthetic snapshot discussed in the chat used **200 frames** and produced:

- **false fault rate:** 0.0%
- **missed fault rate:** 0.0%
- **uncertain rate:** 27.0%
- **legal precision / recall:** 1.000 / 0.940
- **fault precision / recall:** 0.505 / 1.000

## 6.4 Interpretation of those numbers
The synthetic phase was treated as:
- a useful task-definition phase,
- conservative in its use of uncertainty,
- able to catch faults in the toy setup,
- but not something to oversell as real-world performance.

---

# 7. Transition to Real Video

A major shift in the chat happened when attention moved from the synthetic baseline to real side-view footage.

The user explicitly raised the difficulty of the real-world setup:
- the lines were hard to see due to lighting,
- the court could be a tennis court with overlapping markings,
- the camera had small motion,
- and the problem was not simply “see a foot,” but “see a foot relative to the correct boundary at the correct volley moment.”

This led to the conclusion that the real problem was first and foremost a **court registration / geometry problem**, not just a foot-detection problem.

---

# 8. Phase 1: Real Court Registration

## 8.1 First registration assumption and failure
A major lesson established in the chat was that the early simple registration idea failed because it often locked onto the **wrong horizontal line**.

This happened because:
- tennis and pickleball markings overlapped,
- lighting and shadows confused line detection,
- the naive approach treated the task as “find a strong horizontal line,”
- and slight camera motion caused drift.

The user explicitly noted that the current orange line / moving line in the overlay was wrong and not actually the kitchen line.

## 8.2 New framing of the problem
The chat reframed the real problem as:
- **court-structure registration first**,
- then **infer kitchen/NVZ lines from tracked court geometry**,
- then only after that proceed to ball and foot logic.

The net was discussed as a helpful reference, but not sufficient by itself. The more robust framing became:

**anchor-point court geometry + motion stabilization**

rather than

**global line picking**.

## 8.3 Files explicitly described as built in this phase
At one point the chat included a detailed “What was built” summary with a specific commit and file list. That summary included:

### `scripts/`
- `extract_frames.py` — extract at target FPS, save `manifest.csv`
- `annotate_reference.py` — OpenCV click-to-annotate GUI

### `src/`
- `court_registration.py`
- `viz.py`
- `foot_localizer.py` — placeholder at that stage
- `event_detector.py` — placeholder at that stage

### `experiments/`
- `run_court_registration.py`
- `configs/court_reg_v1.yaml`

### `data/real/annotations/`
- `annotations.json`
- `annotations_template.json`
- reference frame JPGs

This phase also produced:
- `line_params.csv`
- `summary_report.json`
- debug frames
- `overlay.mp4`

## 8.4 Commands explicitly discussed in the chat
The following command types were discussed and used in the chat:

### Extract frames
```bash
python scripts/extract_frames.py \
    --video data/real/videos/pickle_vid_1.MOV \
    --out data/real/frames/ \
    --fps 5
```

### Annotate reference frame
```bash
python scripts/annotate_reference.py \
    --video data/real/videos/pickle_vid_1.MOV \
    --frame 60 \
    --out data/real/annotations/annotations.json
```

### Run court registration
```bash
python experiments/run_court_registration.py \
    --config experiments/configs/court_reg_v1.yaml
```

These commands were important because the chat kept returning to the question: *What exactly should I do next inside the repo?*

---

# 9. Court Registration Iterations and Final Strong Result

The chat eventually emphasized a later, stronger registration result, referred to as **court registration v3**.

## 9.1 Final registration method emphasized in chat
The stronger current method was summarized as:

**anchor-point court model + ORB post-translation**

This replaced the earlier fragile line-picking logic.

## 9.2 Key registration numbers used throughout the later chat
The later project summary repeatedly emphasized these numbers:

- **resolution:** 1920 × 1080
- **fps:** 59.943
- **total frames:** 2055
- **duration:** 34.28 s
- **registration success:** 2055 / 2055
- **fallbacks:** 0
- **fallback rate:** 0.0

## 9.3 Comparison result emphasized
The chat also used a direct comparison to show why the chosen registration method mattered:

- **post_translation | fixed | refine-off:** 2055 ok, 0 fallback
- **affine | fixed | refine-off:** 2036 ok, 19 fallback

This became one of the most presentation-safe “hard numbers” in the project.

## 9.4 Why this stage mattered so much
The chat repeatedly returned to the idea that court registration was the **strongest technical win** because:
- if the boundary is wrong, foot decisions are meaningless,
- court geometry is the foundation of the whole pipeline,
- and this stage was the most stable and defensible result on real footage.

---

# 10. Event Timing, Bounce Logic, and Active-Side Inference

After registration, the next major topic in the chat was how to reason about the ball and event timing.

## 10.1 Key logic shift
The chat explicitly moved away from a simplistic “always check if the foot is over the line” framing.

Instead, it converged on this logic:

1. **Track the ball**
2. **Infer bounce vs volley / event timing**
3. **Infer which side/player is active**
4. **Only then check foot relative to the correct NVZ boundary**

This was treated as more faithful to the actual rule and more technically sound.

## 10.2 Bounce logic discussed in the chat
The simplest bounce intuition discussed was:

**ball goes down → near-court turning point → ball goes back up**

The chat then refined that into a more robust bounce candidate definition requiring:
- a local y reversal,
- near-court location,
- enough local trajectory continuity,
- and rejection of junk reacquisition or pickup motions.

## 10.3 Concern about fake bounce candidates
A major issue discussed was that the bounce candidate stage was either:
- too permissive and emitted junk candidates from broken tracks,
- or too strict and emitted zero candidates.

The conclusion was to target a **middle ground**:
- a small believable set of candidate bounces,
- rather than a flood of junk or zero events.

## 10.4 Active-side inference evolution
A specific design pivot discussed in chat was moving from a **single-frame / single-sample side guess** to a **temporal ball-window vote**.

This mattered because the user explicitly noticed that one-frame logic was not reliable.

The later event reasoning story used the following concrete frame examples:

- **frame 929** — improved / flipped correctly to right
- **frame 1537** — improved / flipped correctly to right
- **frame 1948** — remained ambiguous

This became part of both the technical framing and the presentation narrative.

---

# 11. Ball Tracking Questions and Model Strategy

## 11.1 Core question the user asked
A major question was whether the system should use something more advanced than simple classical CV for the ball, especially for **bounce detection**.

## 11.2 Design decision reached in chat
The conclusion reached in chat was:

**For the ball, detection + tracking + trajectory logic is generally better than full segmentation.**

The reasoning was:
- the pickleball is tiny,
- bounce is mostly a **trajectory problem**,
- segmentation is usually not the best tool for such a small fast object,
- and a detector/tracker keeps the bounce logic grounded in real detections.

## 11.3 Ranking given in chat
A ranked recommendation was given in the chat for bounce detection:

1. **Custom small-ball detector + tracker + bounce logic**
2. **Detection + tracker + smoothing / Kalman / raw-point-anchored bounce**
3. **Promptable video segmentation such as SAM-family methods**
4. **Pure classical blob / HSV tracking**

The top choice was explicitly:

**custom small-ball detector + tracker + bounce logic**

while keeping the current pipeline structure rather than switching the whole project over to ball segmentation.

## 11.4 Verification emphasis
A key principle established in the chat was that the pipeline should export review artifacts so the user could ask:
- Is this actually the ball?
- Is this actually a bounce?

rather than blindly trusting a final label.

---

# 12. Foot Localization Questions and Design Decisions

This was another major area of discussion in the chat.

## 12.1 Original concern
The user repeatedly noticed that lower-body / blob-style logic was still too coarse, especially in difficult frames such as frame 1948.

Specific problems included:
- boxes including leg + paddle rather than the shoe alone,
- contact points looking off,
- active-side confusion,
- and overlays showing too much extra geometry.

## 12.2 Stronger idea proposed in chat
A simpler and cleaner upgrade direction emerged:

**Use a pretrained person/pose/segmentation model to focus only on the active player’s lower body / feet.**

The logic was:
- first isolate the person,
- then focus on lower-body / foot region,
- then compute the foot contact point,
- then compare it to the registered boundary.

This was treated as much cleaner than letting paddle/hand/blob heuristics dominate.

## 12.3 Important scoped decision
The chat explicitly advised **not** to replace the whole pipeline.

Instead, the recommended upgrade was:

**keep the current pipeline, but strengthen only the foot localization stage**

This was the “surgical upgrade” strategy:
- keep registration,
- keep event logic,
- keep signed-distance fault rule,
- improve foot localization on candidate frames.

## 12.4 Foot decision thresholds used in the presentation/report framing
The final threshold logic repeatedly used in the chat was:

- **d > +15 px** → `legal_volley`
- **-5 px <= d <= +15 px** → `uncertain`
- **d < -5 px** → `foot_fault_volley`

The interpretation emphasized in chat was:
- positive = legal side,
- negative = inside the kitchen,
- near boundary = uncertain.

---

# 13. Phase 2: End-to-End Demo Pipeline

The later part of the chat treated the project as an end-to-end demo pipeline rather than disconnected pieces.

## 13.1 Final pipeline structure used in the chat
The pipeline settled into this sequence:

**Video → Court registration → Ball tracking → Bounce / volley inference → Foot localization → Signed distance → legal / fault / uncertain**

Human review and overrides were explicitly treated as part of the workflow, not a hidden embarrassment.

## 13.2 Demo summary numbers used repeatedly
The chat used the following demo summary snapshot:

- **3 detected event frames**
- **1 likely foot_fault_volley**
- **2 uncertain**

## 13.3 Event table values that were explicitly referenced
The demo event CSV values discussed in chat included:

### Frame 929
- timestamp ≈ 15.50 s
- active side: right
- label: `foot_fault_volley`
- signed distance ≈ **-10.78 px**
- foot mode: `event_hybrid`

### Frame 1537
- timestamp ≈ 25.64 s
- active side: right
- label: `uncertain`
- signed distance ≈ **-41.93 px** for the chosen side, with review required
- lower confidence / ambiguity

### Frame 1948
- timestamp ≈ 32.50 s
- active side: left in one run, but confidence only ≈ **0.551**
- label: `uncertain`
- one interpretation used `left_signed_dist_px = 21.72` and `right_signed_dist_px ≈ 0.02`
- the system repeatedly treated this as a hard ambiguous case

## 13.4 How the chat framed these outputs
The end-to-end demo was repeatedly framed as:
- **wired and producing outputs**,
- but still fragile in the ball/event stage,
- and best presented as a **working research prototype with human review**, not a finished automatic referee.

---

# 14. Human-in-the-Loop Review Architecture

One of the strongest conceptual decisions made in the chat was to insist that the system should **not silently guess** in ambiguous cases.

## 14.1 Review checkpoints discussed
The pipeline was described as needing checkpoints where the user could validate:
- court / line correctness,
- ball tracking correctness,
- bounce candidate correctness,
- foot point correctness,
- active side correctness,
- final event label correctness.

## 14.2 The purpose of this review loop
The chat framed this as a strength:
- it makes the system more honest,
- it avoids false confidence,
- and it better matches a research-demo workflow.

This became central to both the technical story and the presentation story.

---

# 15. Presentation Work Done in the Chat

A very large part of the later conversation focused on building the presentation.

## 15.1 Slide story that was developed
The presentation eventually converged on a 10-slide story:

1. Title
2. Problem and Motivation
3. Research Questions
4. Project Evolution and Design Lessons
5. End-to-End Pipeline
6. Deep Dive: Court Registration
7. Deep Dive: Event Timing and Side Inference
8. Deep Dive: Foot Localization and Decision Logic
9. Current Results and Limitations
10. Conclusion and Next Steps

## 15.2 Strong presentation advice developed in chat
The deck was repeatedly guided to emphasize:
- the project evolution,
- failed approaches and design pivots,
- court registration as the clearest technical win,
- honest prototype framing,
- uncertainty / human review as a deliberate design choice,
- and not overselling final detection accuracy.

## 15.3 Slide assets created / discussed
During the chat, a large number of presentation artifacts were created or used, including:

- `01_registration_overlay.png`
- `02_registration_comparison.png`
- `03_detected_fault_event.png`
- `04_uncertain_event.png`
- `05_uncertain_review_event.png`
- multiple generated slide image mocks
- PowerPoint versions:
  - `KitchenMaster_Presentation.pptx`
  - `KitchenMaster_Presentation_simpler.pptx`
  - `KitchenMaster_Presentation_revised.pptx`

## 15.4 Design feedback given by the user
The user explicitly disliked parts of the deck that felt too AI-generated, including:
- overly decorative metric cards,
- too many green checkmarks / icons,
- overly polished infographic elements,
- ambiguous AI-looking pipeline blocks,
- and too much visual styling around text summaries.

The later slide revision instructions therefore stressed:
- simplify slide 5, 6, 7, 9, and 10,
- use normal bullets / text boxes,
- remove too many card/checkmark UI elements,
- keep it modern but more like a real student engineering presentation.

## 15.5 Prompt work for Cursor / Claude
The chat also produced detailed prompts for Cursor / Claude Code to:
- inspect the existing repo before changing anything,
- build YAML-driven pipelines,
- add review checkpoints,
- upgrade foot localization,
- and remake slides in a less AI-looking style.

---

# 16. Concrete Design Pivots Captured in the Chat

These are the major design/engineering lessons that came out of the project discussion.

## 16.1 “Simple filtering was not enough”
This was one of the main lessons.

Why:
- wrong lines were selected,
- overlapping tennis/pickleball markings confused detection,
- camera motion broke static assumptions,
- and court geometry needed to be modeled more explicitly.

## 16.2 “Court registration is a geometry problem first”
The chat moved the project away from “find a line” and toward:
- anchor-point court geometry,
- tracked net/court structure,
- and propagated boundaries over time.

## 16.3 “Temporal context beats one-frame sampling”
This came up especially in:
- bounce logic,
- active-side inference,
- and deciding whether to trust an event.

## 16.4 “Uncertain is better than wrong confidence”
This became one of the most consistent themes in the project and presentation.

## 16.5 “Keep the pipeline modular”
The chat repeatedly argued against throwing away everything and building one giant end-to-end stack. Instead it emphasized:
- strengthen weak modules,
- keep the interpretable pipeline,
- and improve individual stages such as foot localization or ball detection.

---

# 17. What the Assistant Did During This Chat

This section summarizes what was done in response to the user’s questions.

## 17.1 Research / problem-planning support
The assistant:
- helped frame the problem as a real research plan,
- emphasized the importance of the user owning the scientific judgment,
- suggested dividing work between human judgment and agent/code assistance,
- and helped transform the project into a staged research pipeline.

## 17.2 Technical diagnosis of failures
The assistant repeatedly:
- analyzed why early line picking failed,
- diagnosed that the overlay lines were wrong,
- explained why registration drift was happening,
- and identified court registration as the true Phase 1 bottleneck.

## 17.3 Proposed repo next steps
The assistant generated multiple targeted prompts for Cursor / Claude Code to:
- validate Phase 1,
- move from simple line detection to court-structure registration,
- add YAML-driven demo pipelines,
- add review checkpoints,
- improve bounce logic,
- improve foot localization,
- and preserve modularity.

## 17.4 Presentation guidance
The assistant:
- proposed multiple versions of the slide structure,
- turned project context into a research presentation story,
- emphasized design pivots and lessons learned,
- helped develop speaker framing,
- and generated presentation artifacts / drafts.

## 17.5 Honest scoping under time pressure
When the user mentioned having only a few hours before presentation, the assistant repeatedly shifted the goal from “solve everything” to:
- a believable feasibility demo,
- ball tracking overlays,
- a small curated set of event clips,
- and honest prototype framing.

---

# 18. Current Project State at the End of This Chat

By the end of this chat, the project should be understood as:

## Strongest current result
**Court registration v3**
- anchor-point court model + ORB post-translation
- 2055 / 2055 registered frames
- 0 fallbacks
- stronger than the compared affine setup on the demo clip

## Full pipeline status
The end-to-end demo pipeline exists and produces outputs, but is still fragile.

## Weakest stage
The weakest stage is the **ball/event reasoning stage**, followed by foot localization on hard frames.

## Labeling philosophy
The project intentionally preserves **uncertainty** rather than forcing all cases into legal/fault.

## Current demo snapshot
- 3 event frames
- 1 likely foot fault
- 2 uncertain

## Overall framing
This is best presented and reported as a **working research prototype with human review**, not as a finished production referee.

---

# 19. Best Report / Presentation-Safe Main Claims

If this final report is turned into prose, the strongest defensible claims from the chat are:

1. **A single side-view camera can support interpretable foot-fault analysis, but only when court geometry is reliably registered first.**
2. **Court registration became the strongest technical result of the project.**
3. **The project evolved meaningfully through failed simpler approaches, especially simple filtering and naive line selection.**
4. **Temporal context improved event-side inference and helped avoid one-frame mistakes.**
5. **Uncertainty and human review are deliberate design choices that reduce obviously wrong confident calls.**
6. **The full pipeline is wired end-to-end, but the ball/event stage remains the largest remaining bottleneck.**

---

# 20. Remaining Work Identified in the Chat

The chat repeatedly pointed to the following next steps:

- improve ball detector
- improve bounce / volley inference
- gather and label more real clips
- strengthen foot localization with better pose / segmentation methods
- reduce dependence on manual overrides
- quantify override frequency
- study sim-to-real transfer more systematically
- validate the system more rigorously on real data

---

# 21. Suggested Structure for the Final Written Report

If this Markdown is converted into the actual final report, a strong report structure would be:

1. Introduction / Motivation
2. Research Questions
3. Related Framing / Problem Definition
4. System Overview
5. Synthetic Baseline
6. Real Court Registration
7. End-to-End Demo Pipeline
8. Design Pivots and Lessons Learned
9. Results
10. Limitations
11. Human-in-the-Loop Review and Uncertainty
12. Future Work
13. Conclusion

---

# 22. Short Executive Summary

KitchenMaster is a research prototype for side-view pickleball NVZ foot-fault analysis. The project was developed in three layers: a synthetic baseline to define the task and evaluation logic, a real court-registration stage to estimate court geometry from video, and an end-to-end demo pipeline connecting registration, ball reasoning, foot localization, signed-distance classification, and human review. The strongest technical result reached in the chat was court registration v3, which achieved 2055/2055 registered frames with 0 fallbacks on the main demo clip. The end-to-end pipeline currently produces 3 event outputs, of which 1 is a likely foot fault and 2 are uncertain. The project’s central design principle is that uncertain is preferable to a wrong confident call, and the final system is best understood as a modular, interpretable, human-in-the-loop research prototype rather than a finished automatic officiating product.

---

# 23. Appendix: Notable Files Mentioned in This Chat

## Code / pipeline files
- `scripts/extract_frames.py`
- `scripts/annotate_reference.py`
- `src/court_registration.py`
- `src/court_model.py`
- `src/stabilizer.py`
- `src/viz.py`
- `src/ball_tracker.py`
- `src/volley_classifier.py`
- `src/foot_localizer.py`
- `src/foot_fault_pipeline.py`
- `experiments/run_court_registration.py`
- `experiments/run_court_registration_v3.py`
- `experiments/run_demo_pipeline.py`
- `experiments/configs/court_reg_v1.yaml`
- `experiments/configs/demo_pipeline.yaml`

## Output / evidence files
- registration overlays and debug frames
- `summary_report.json`
- `ball_tracking.csv`
- `candidates.csv`
- `events.csv`
- `foot_fault_events.csv`
- `review_pending.json`
- `review_approved.json`

## Slide / presentation files created in chat
- `KitchenMaster_Presentation.pptx`
- `KitchenMaster_Presentation_simpler.pptx`
- `KitchenMaster_Presentation_revised.pptx`
- registration slide assets
- event frame slide assets
- generated slide preview images

---

# 24. Appendix: One-Sentence Version

KitchenMaster evolved from a synthetic classical-CV baseline into a real side-view court-registration and foot-fault analysis research pipeline, where the clearest technical win was reliable court registration and the clearest design lesson was that uncertainty and human review are necessary when event timing and foot localization remain ambiguous.
