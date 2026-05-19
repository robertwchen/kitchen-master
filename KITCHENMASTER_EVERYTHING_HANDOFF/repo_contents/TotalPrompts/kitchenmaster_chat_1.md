# KitchenMaster Project Report From This Chat

## Purpose of this document
This markdown file is a thorough reconstruction of what was discussed in this chat about the **Sensors and Ubiquitous Computing** project, what the project scope became, what planning decisions were made, what implementation milestones were defined, what outputs were produced, what the current repo state appears to be, and what the remaining work is.

This is **not** a literal transcript of every message. It is a structured report of the project work, decisions, prompts, deliverables, metrics, and reasoning that were developed in the conversation.

---

## Project identity
- **Project name:** KitchenMaster
- **Course:** ECE 4501/6501 Sensors and Ubiquitous Computing
- **Student:** Robert Chen
- **Core project idea:** Detect pickleball kitchen / non-volley-zone (NVZ) foot faults from a portable fixed side-view camera.
- **Original framing from the proposal:** build a portable clip-on net-post device that detects NVZ line contact during volleys and gives immediate feedback plus logging.

### Proposal grounding used in this chat
The project proposal established the following:
- Pickleball kitchen foot faults are contentious and frequently self-officiated.
- Existing high-end systems are accurate but expensive and non-portable.
- The project research questions centered on:
  - **RQ1 Detection:** can a portable net-mounted or side-mounted setup reliably detect NVZ line contact?
  - **RQ2 Robustness:** how do lighting, mount location, occlusion, and court variation affect performance?
  - **RQ3 Decision quality:** can confidence and an **uncertain** output reduce wrong calls while keeping trust high?
- Success criteria in the proposal were already about controlled detection performance, low false beeps, fast setup, and clear failure modes.

---

## What you originally wanted
You said, in substance:
- you had a whole research project presentation but had not started the project
- you wanted Codex or Claude Code to do most of the research, testing, building, graph generation, classification checks, and repo work
- you wanted the work to look like a real graduate-level research project
- you wanted to start with something like a pickleball game engine or simulation environment
- you wanted side-profile detection of the court kitchen line, foot position, and eventually ball behavior
- you wanted the project to become a clean GitHub repo you could use on a resume
- you wanted the code and structure to feel understandable and not overly AI-looking

That request triggered the main planning work in this chat.

---

## The main strategic answer given in this chat
The project was split into **three lanes**.

### Lane 1: You
You were told to personally own the actual research judgment:
- final project scope
- what exactly counts as a kitchen violation
- labeling rules
- evaluation decisions
- what results are honest enough to present

Reason given:
If Codex built everything and you could not explain the choices, the repo might look polished but the project would feel fake in a demo or Q&A.

### Lane 2: ChatGPT
I took the role of **research lead / project manager**:
- turning the proposal into a real research plan
- defining milestones, ablations, metrics, and baselines
- designing repo structure
- writing the experiment plan
- deciding what to test first
- interpreting results
- turning outputs into slides and report language
- writing tight Claude/Codex prompts

### Lane 3: Codex / Claude Code
These tools were assigned the implementation-worker role:
- initialize repo
- create folder structure
- set up Python environment, tests, CI or near-CI organization
- build synthetic environment
- build data pipeline
- write baseline detection code
- save plots and metrics
- generate README and docs
- make focused commits

The key warning was:
**Do not ask Codex to “do all the research.”**
It can code. It cannot replace experiment design, ground-truth decisions, and scientific honesty.

---

## The major scope correction made in this chat
At first, there was discussion of using a more realistic game engine or simulation environment. That idea was not rejected completely, but it was **deprioritized**.

### Why the scope was corrected
Because of the short timeline, the project could not afford to spend most of its time building a polished sports simulator.

The corrected principle was:
- the core research question is **not** “can I build a realistic pickleball game engine?”
- the real question is **“can I detect foot-line contact reliably enough to study the problem?”**

That shifted the plan toward:
1. minimal synthetic environment first
2. interpretable baseline detector
3. small real dataset
4. evaluation and failure analysis
5. slides and final report

This was explicitly framed as consistent with the class expectations for:
- context refresher
- research questions
- what you built
- data collected
- preliminary results
- next steps

---

## The 3-day emergency plan that was created
Once you said you had only 3 days, the plan was compressed into a survival workflow.

### Day 1 goal
Create the repo, define labels, build a synthetic environment, and generate first outputs.

Concrete actions proposed:
- create repo and folder structure
- define the 4 scenario classes
- define the 3 output labels
- fix a single camera setup
- send two initial Claude Code prompts:
  1. repo scaffold prompt
  2. synthetic data pipeline prompt

### Day 2 goal
Build the baseline detector and collect a tiny real dataset.

Concrete actions proposed:
- implement line detection + foot localization + overlap logic
- generate predictions, confusion matrix, precision, recall, uncertain rate
- record a small real dataset from the intended camera angle
- run the pipeline on real clips

### Day 3 goal
Finalize results, slides, and report.

Suggested deliverables:
- system diagram
- synthetic sample frames
- dataset summary table
- confusion matrix
- precision / recall summary
- failure-case slide
- next-steps slide

### Slide structure recommended during the chat
A 12-slide storyline was suggested:
1. Big context
2. Gap / related work
3. Research questions
4. System overview
5. Synthetic environment
6. Real data collection setup
7. Baseline method
8. Dataset summary
9. Preliminary results
10. Failure cases
11. Next steps
12. Thank you

---

## The exact technical setup plan that was written in this chat
A concrete repo setup walkthrough was produced.

### Proposed local setup commands
```bash
mkdir kitchen-master
cd kitchen-master
git init
mkdir -p docs data/sim data/real src experiments results tests
touch README.md .gitignore
```

### Suggested `.gitignore`
```gitignore
__pycache__/
*.pyc
.venv/
venv/
.env
.DS_Store
results/**/*.png
results/**/*.mp4
results/**/*.csv
data/real/raw/*
data/real/temp/*
data/sim/raw/*
```

### Suggested Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib opencv-python pyyaml scikit-learn pytest
```

### Suggested initial files
```text
docs/problem.md
docs/plan.md
src/__init__.py
src/config.py
src/sim_generator.py
src/baseline_detector.py
src/evaluate.py
experiments/run_sim.py
experiments/run_eval.py
```

### Suggested GitHub connection step
```bash
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git add .
git commit -m "Initial research repo scaffold"
```

---

## The core problem definition created in this chat
A markdown problem statement was drafted for `docs/problem.md`.

### Project goal
Build a first-pass portable side-view vision prototype for pickleball kitchen / NVZ foot-fault detection.

### Core detection task
Given a fixed side-view camera near the kitchen line, determine whether a player’s foot:
1. stays legal behind the line
2. touches or crosses the line and commits a fault
3. cannot be determined reliably and should be marked uncertain

### Research questions formalized in this chat
- **RQ1:** Can a fixed side-view portable camera detect NVZ line contact in controlled conditions?
- **RQ2:** How sensitive is detection to viewpoint, blur, occlusion, and line-foot distance?
- **RQ3:** Can an uncertain output reduce wrong calls in ambiguous cases?

### Initial scope explicitly chosen
- one camera only
- one court region only
- side-view profile only
- controlled clips first
- synthetic data first, then small real dataset

---

## The 4 scenarios and 3 labels defined in this chat
These were a major part of the project formalization.

### Scenario classes
1. **Clear legal**  
   The foot remains completely behind the kitchen line during the volley event window.

2. **Clear fault**  
   The foot clearly touches or crosses the kitchen line during the volley event window.

3. **Borderline contact**  
   The foot comes within a tiny margin of the line or appears to barely touch it, making the decision sensitive to resolution, blur, or calibration.

4. **Occluded or uncertain**  
   The foot-line relationship cannot be determined reliably because of occlusion, motion blur, bad lighting, or partial visibility.

### Output labels
- `legal`
- `fault`
- `uncertain`

These classes and labels became the backbone of the synthetic generator, baseline classifier, and presentation story.

---

## The metrics and evaluation plan created in this chat
The evaluation framework proposed in the conversation included:
- confusion matrix
- precision
- recall
- uncertain rate
- number of false fault calls
- number of missed faults

The project framing explicitly favored **interpretable, saved, reproducible outputs** instead of flashy claims.

---

## The initial Claude Code prompt sequence written in this chat
A full prompt sequence was drafted.

### Prompt 1: repo scaffold
This prompt told Claude Code to:
- create a clean research-first GitHub repo
- use Python only
- keep the code simple, readable, and modular
- create the core folders
- add a concise README
- add config-driven experiment support
- add logging and metric saving
- add plot saving
- avoid fabricated data, metrics, graphs, and conclusions
- prefer interpretable baselines over heavy models
- create placeholders or stubs for synthetic generation, baseline detection, evaluation, and experiment scripts

### Prompt 2: minimum viable synthetic data pipeline
This prompt told Claude Code to:
- build a simple controllable side-view simulation in Python
- render the kitchen line clearly
- represent the foot with a simple proxy shape
- generate labeled sequences for the 4 scenario types
- save outputs under `data/sim/`
- export image sequences or videos, metadata CSV, labels, and useful parameters like foot position, line position, occlusion flag, blur level
- make the generator configurable
- save sample visualizations to `results/`

### Prompt 3: baseline detector
This prompt told Claude Code to:
- detect the kitchen line
- detect or localize the foot proxy
- classify each case as legal, fault, or uncertain
- use a simple rule-based or geometric method first
- add confidence logic so ambiguous cases can become uncertain
- save predictions to CSV
- compute confusion matrix, precision, recall, and uncertain rate
- save plots and visual overlays

### Prompt 4: real data extension
This prompt told Claude Code to:
- support a small real dataset from a fixed side-view camera
- expect videos or frames under `data/real/`
- use an annotation CSV with `clip_name, ground_truth_label, notes`
- reuse as much of the baseline pipeline as possible
- save per-clip predictions, confidence scores, and overlays
- save real-data metrics separately
- document failure modes like blur, shoe color, occlusion, and calibration drift

---

## What repo and file outputs were later shown in this chat
After the first two prompts were run, the following repo tree was shared:

```text
kitchen-master/
├── .gitignore
├── README.md
├── requirements.txt
├── docs/
│   ├── plan.md
│   └── problem.md
├── experiments/
│   ├── configs/
│   │   └── sim_v1.yaml
│   ├── run_eval.py
│   └── run_sim.py
├── src/
│   ├── __init__.py
│   ├── baseline_detector.py
│   ├── config.py
│   ├── evaluate.py
│   └── sim_generator.py
└── tests/
    ├── __init__.py
    ├── test_detector.py
    ├── test_evaluate.py
    └── test_sim_generator.py
```

### Synthetic-generation-related files reported
```text
src/sim_generator.py
experiments/run_sim.py
experiments/configs/sim_v1.yaml
tests/test_sim_generator.py
src/config.py
src/evaluate.py
experiments/run_eval.py
```

---

## README contents reported back in the chat
The generated README described KitchenMaster as:
- a research prototype for pickleball NVZ foot-fault detection from a fixed side-view camera
- structured around the three research questions above
- using the three output labels legal, fault, uncertain
- organized into docs, data, src, experiments, results, and tests
- supporting commands such as:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python experiments/run_sim.py
python experiments/run_sim.py --config experiments/configs/sim_v1.yaml
python experiments/run_eval.py --results results/sim_v1/
pytest tests/
```

The README also stated that outputs such as metrics and confusion matrices are saved under `results/<run_name>/`.

---

## The first metrics that were shown in this chat
A metrics CSV example was surfaced with the following values:

```text
n,precision_legal,recall_legal,precision_fault,recall_fault,precision_uncertain,recall_uncertain,uncertain_rate,false_fault_rate,missed_fault_rate
200,1.0,0.9,0.5319,1.0,0.918,0.56,0.305,0.0,0.0
```

### Interpretation that was given in the chat
The interpretation offered was:
- the model was high-recall on **fault** detection
- it almost never missed faults
- it overcalled fault in ambiguous cases
- uncertainty handling was still weak

That led to a suggested honest slide sentence:
> “Our first synthetic baseline is high-recall for fault detection but tends to overcall faults in ambiguous cases, which motivates better uncertainty gating and calibration.”

This sentence was specifically presented as a good research-style claim because it is honest, concrete, and interpretable.

---

## The concern that came up about metadata
After seeing the early outputs, an issue was identified:
- the current repo did not appear to generate a full canonical metadata CSV at first
- `predictions.csv` was being used as the closest artifact in one export note

That led to a suggestion to improve the research quality by saving richer per-sample metadata such as:
- sample id
- scenario type
- ground-truth label
- foot parameters
- line position
- signed distance to line
- occlusion flag
- blur level
- seed
- output file path
- notes

The reason was not to change the project, but to make the synthetic pipeline produce evidence that would be more useful for failure analysis and final reporting.

---

## The clarification made later in the chat: the metadata and more advanced outputs were already done
You later explained that the repo had already advanced much further than the earlier shared artifacts suggested, and that everything was already live on GitHub at commit:
- **`c978c6c`**

### You said the following were already complete
- repo structure preserved, with original files intact
- `SampleMeta` dataclass implemented with:
  - `sample_id`
  - `scenario_type`
  - `ground_truth_label`
  - `foot_x`
  - `foot_y`
  - `foot_width`
  - `foot_height`
  - `line_y`
  - `signed_distance_px`
  - `occlusion_flag`
  - `blur_level`
  - `seed`
  - `frame_path`
- metadata written to:
  - `results/sim_v1/metadata.csv`
- baseline detector completed with:
  - Hough line detection
  - HSV foot segmentation
  - margin-based classification
  - `predict_with_details`
- saved outputs included:
  - `predictions.csv`
  - `metrics.csv`
  - `confusion_matrix.csv`
  - `confusion_matrix.png`
  - **21 overlay PNGs** in `results/sim_v1/overlays/`
- `failure_analysis.csv` already generated with:
  - **12 groups** by scenario × occlusion × blur × distance bucket
- real-data workflow already created:
  - `data/real/videos/`
  - `data/real/frames/`
  - `data/real/annotations.csv` template
  - `experiments/run_real.py`
- README and `docs/plan.md` already updated with:
  - status table
  - real results
  - run commands

This was a major status update in the conversation because it confirmed that the repo had progressed all the way through the originally planned synthetic + baseline + evaluation + real-data pipeline setup.

---

## The real-data command that was present by the end of the chat
You said that what was actually left for Day 2 was:
- collect real clips
- extract frames into `data/real/frames/`
- fill in `data/real/annotations.csv`
- run:

```bash
python experiments/run_real.py \
  --annotations data/real/annotations.csv \
  --results results/real_v1/
```

That means the code-side work had reached the point where the next main bottleneck was **data collection**, not more architecture or simulation.

---

## What was said about changing direction
At one point you asked why the advice seemed to shift.

The clarification given was:
- the overall plan had **not** changed
- you had already completed Phase 1 and Phase 2
- the later prompts were meant to move from “build the skeleton and first prototype” to “turn the prototype into research evidence”
- the later emphasis on metadata, failure analysis, and real-data support was not a pivot, only the next stage of the same project

This distinction mattered because the conversation repeatedly avoided restarting or redesigning the project once useful work already existed.

---

## What was finally identified as the true next step
By the end of the exchange, the conclusion was:

### The next step is **not** more simulation
The synthetic environment had already served its purpose.

### The missing piece is real-world validation
Even if small, the project needed some reality check to support the presentation and final report.

The strongest recommendation became:
1. record a tiny real dataset yourself if possible
2. or, quickly search for side-profile videos that are usable
3. do not spend much more time polishing simulator realism

### Suggested minimum real dataset sizes discussed
Several versions were suggested during the chat.

#### Larger small set
- 10 legal
- 10 fault
- 5 borderline
- 5 uncertain

#### Reduced small set
- 6 legal
- 6 fault
- 4 borderline
- 4 uncertain

#### Bare minimum fallback
- 4 legal
- 4 fault
- 2 borderline
- 2 uncertain

The argument was that even **12 to 20 clips** would be better than zero real validation.

---

## What was said about searching for existing videos
You asked whether you could avoid recording and instead find side-profile videos.

The answer developed in the chat was:
- yes, try briefly to search for them
- no, do not waste too much time on the search
- do not rely on there being a perfect public dataset
- if a quick search fails, record your own small set immediately

### What was found in that discussion
It was noted that:
- some public pickleball computer-vision resources exist for court lines or general ball/court work
- some YouTube videos exist explaining kitchen violations
- but there did not appear to be a clearly established public dataset specifically for **side-view kitchen foot-fault clips labeled legal / fault / uncertain**

That reinforced the idea that your own small fixed-view dataset would still be the most defensible and controllable validation source.

---

## The final recommended workflow by the end of the chat
By the end, the most coherent project workflow had become:

### Phase 0
Repo scaffold

### Phase 1
Define exact scenarios and labels

### Phase 2
Build synthetic side-view generator

### Phase 3
Build interpretable baseline detector

### Phase 4
Add evaluation outputs and failure analysis

### Phase 5
Add real video ingestion and annotation workflow

### Phase 6
Collect a tiny real dataset and run `run_real.py`

### Phase 7
Build final presentation and final report around:
- synthetic development
- interpretable baseline
- preliminary real validation
- failure modes
- next steps

---

## The presentation story that emerged from this chat
A strong and honest narrative was shaped during the conversation.

### Suggested claim style
Do **not** claim:
- that the system solved kitchen violations generally
- that it is production-ready
- that it performs robustly under all conditions

Do claim something like:
- this is a first-pass portable side-view prototype
- it was developed in controlled synthetic conditions
- it was evaluated with an interpretable baseline
- preliminary real validation was conducted on a small controlled set
- ambiguity handling and uncertainty remain central challenges

### Example one-sentence story
> We built a first-pass portable side-view prototype for pickleball NVZ line-contact detection, developed and tested it in a controllable synthetic environment, and performed preliminary real-world validation on a small clip set to study both detection performance and failure modes.

That is the most faithful summary of the project framing that came out of this chat.

---

## The code and implementation details explicitly mentioned in this chat
The following technical items were explicitly discussed and should be recorded in the final report.

### Repo organization
```text
docs/
sim/ or data/sim/
data/
src/vision/
src/rules/
experiments/
results/plots/
README.md
AGENTS.md (suggested in planning)
```

The actually reported repo tree used:
```text
docs/
data/sim/
data/real/
src/
experiments/
results/
tests/
```

### Synthetic generator concepts
- simple controllable side-view scene
- court line rendered clearly
- simple foot proxy instead of full game realism
- saved labels and metadata
- deterministic seeds where reasonable
- configurable experiment script using YAML

### Baseline detector concepts
- Hough line detection
- HSV foot segmentation
- rule-based geometric classification
- uncertainty thresholding / margin logic
- `predict_with_details`

### Evaluation artifacts discussed
- `predictions.csv`
- `metrics.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `failure_analysis.csv`
- overlay image examples
- per-class precision / recall plot suggested

### Real-data workflow concepts
- `data/real/videos/`
- `data/real/frames/`
- `data/real/annotations.csv`
- `experiments/run_real.py`
- manual labeling of clips
- fixed side-view camera

---

## The commit and status milestone explicitly mentioned
You reported that all the listed tasks were already done and live on GitHub at:
- **commit `c978c6c`**

That commit should be included in any final report or appendix as the checkpoint representing the first full synthetic-plus-real-workflow prototype state.

---

## Practical conclusions that this chat established
1. The project was successfully narrowed from a broad “let AI do the whole research project” idea into a manageable research prototype.
2. The scientifically important elements were defined by you and with my help, not left fully to Codex.
3. The project was intentionally kept interpretable and simple.
4. The side-view synthetic pipeline was used as a development tool, not as the final evidence source.
5. Real-world validation, even if tiny, became the decisive next step.
6. The most honest presentation angle is a **controlled preliminary prototype**, not a finished officiating product.

---

## A concise chronology of the chat
### Stage A: initial request
You asked how to have Codex / Claude build the whole research project, including research, testing, graphs, and repo setup.

### Stage B: strategic split
The 3-lane model was introduced: you, ChatGPT, Codex.

### Stage C: emergency compression
Because of the 3-day deadline, the plan was reduced to a fixed side-view synthetic-to-real prototype pipeline.

### Stage D: exact setup and prompts
A precise repo setup guide, labels, scenarios, and Claude prompt sequence were written.

### Stage E: first repo output review
You shared repo tree, README, file list, and metrics examples from the first generated implementation.

### Stage F: metadata / evidence tightening
I recommended richer metadata and failure analysis when it looked like those outputs might be thin.

### Stage G: clarification that the repo had advanced further
You reported that all of those items were already implemented and live at commit `c978c6c`.

### Stage H: identify true next step
The conversation concluded that the next step was no longer simulation, but real-clip collection and running the real-data pipeline.

---

## What should go into the final course report, based on this chat
If you write a final report from this project, the sections should roughly be:

### 1. Introduction / problem
- why kitchen violations matter in self-officiated pickleball
- why portable systems matter
- why side-view geometry is attractive

### 2. Research questions
Use the RQ1/RQ2/RQ3 wording above.

### 3. System design
- fixed side-view camera assumption
- synthetic scene generator
- baseline detector
- confidence / uncertain output

### 4. Synthetic dataset
- the 4 scenarios
- metadata fields
- controlled perturbations like blur and occlusion
- why synthetic data was useful for early prototyping

### 5. Baseline method
- line detection
- foot segmentation
- rule-based overlap logic
- uncertainty gating

### 6. Evaluation
- confusion matrix
- precision / recall
- false-fault tendency
- uncertain-rate behavior
- failure analysis by scenario, blur, occlusion, and distance bucket

### 7. Real-data workflow
- collected side-view clips or curated clips
- manual annotations
- use of `run_real.py`
- preliminary real-world testing

### 8. Results and discussion
- what worked in controlled conditions
- where ambiguity caused overcalling or uncertain handling problems
- how the synthetic and real results compare

### 9. Limitations
- very small real dataset
- controlled view only
- not robust to all shoes / courts / lighting
- not yet event-complete if full volley logic or player context is limited

### 10. Next steps
- larger real dataset
- better calibration
- temporal reasoning over clips
- potentially richer event logic
- more court / lighting diversity

---

## Suggested appendix content derived from this chat
You could add appendices containing:
- repo tree
- run commands
- config file example
- metadata schema
- sample annotation CSV
- example metrics table
- example confusion matrix
- overlay images with predicted vs true labels
- commit hash `c978c6c`

---

## Short “what ChatGPT did in this chat” section
If you want a section in the final report describing the assistance process, this is the accurate version:

### ChatGPT contributions in this chat
- converted the broad project idea into a concrete research plan
- narrowed the project scope to a feasible 3-day prototype
- defined the side-view problem statement
- formalized the 4 scenario types and 3 output labels
- designed the staged workflow from synthetic generation to real validation
- proposed repo structure and reproducibility practices
- wrote exact Claude Code prompt sequences for scaffold, synthetic generation, baseline detection, and real-data support
- interpreted early synthetic metrics and suggested honest claim language
- helped distinguish between implementation progress and true research evidence
- recommended moving from simulation to real-clip validation once the code pipeline existed

### What you still owned
- the meaning of a kitchen violation
- the labeling decisions
- the choice of which results were honest enough to present
- the eventual real clip collection and validation setup

---

## One-paragraph executive summary
KitchenMaster was developed in this chat from an initially broad idea of using Codex or Claude Code to automate an entire ubiquitous computing class project into a focused, interpretable research prototype for pickleball kitchen foot-fault detection. The final scope centered on a fixed side-view camera, three output labels (legal, fault, uncertain), four scenario classes (clear legal, clear fault, borderline contact, occluded/uncertain), a controllable synthetic data generator, and a rule-based baseline using line detection, foot segmentation, and threshold-based classification. The workflow emphasized reproducibility through saved configs, metrics, metadata, failure analysis, and overlays. Early synthetic results suggested perfect fault recall but lower fault precision and only moderate uncertain recall, supporting an honest narrative that the model is conservative and overcalls faults in ambiguous cases. By the end of the chat, the repo reportedly included a metadata dataclass, confusion matrices, overlays, failure-analysis outputs, and a real-data pipeline live at commit `c978c6c`, making the collection of a small real side-view dataset the primary remaining step before final reporting and presentation.

---

## Quick reference: important numbers and concrete facts from this chat
- **3-day deadline** drove the compressed project plan
- **3 labels:** legal, fault, uncertain
- **4 scenarios:** clear legal, clear fault, borderline contact, occluded/uncertain
- **Reported synthetic evaluation sample count:** `n = 200`
- **Reported synthetic metrics:**
  - precision_legal = `1.0`
  - recall_legal = `0.9`
  - precision_fault = `0.5319`
  - recall_fault = `1.0`
  - precision_uncertain = `0.918`
  - recall_uncertain = `0.56`
  - uncertain_rate = `0.305`
  - false_fault_rate = `0.0`
  - missed_fault_rate = `0.0`
- **Reported overlays saved:** `21` PNGs
- **Reported failure-analysis grouping count:** `12` groups
- **Important reported commit:** `c978c6c`
- **Real-data runner command:**
  ```bash
  python experiments/run_real.py \
    --annotations data/real/annotations.csv \
    --results results/real_v1/
  ```

---

## Final bottom-line takeaway from this chat
The most important outcome of this entire conversation is that the project became **research-shaped** instead of just “AI-generated code.” The scope was narrowed, the labels and metrics were formalized, the synthetic environment was used strategically instead of as an end in itself, the baseline was kept interpretable, and the final missing step was correctly identified as small-scale real-world validation rather than more simulator polish.
