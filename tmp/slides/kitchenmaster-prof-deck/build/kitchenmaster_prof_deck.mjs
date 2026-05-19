const fs = await import("node:fs/promises");
const path = await import("node:path");
const { Presentation, PresentationFile } = await import("@oai/artifact-tool");

const W = 1280;
const H = 720;

const DECK_ID = "kitchenmaster-prof-deck";
const OUT_DIR = "/Users/robert/dev/git/school/kitchen-master/kitchen-master/outputs/slides/kitchenmaster-prof-deck";
const REF_DIR = "/Users/robert/dev/git/school/kitchen-master/kitchen-master/photos_for_slides";
const SCRATCH_DIR = path.resolve(process.env.PPTX_SCRATCH_DIR || path.join("tmp", "slides", DECK_ID));
const PREVIEW_DIR = path.join(SCRATCH_DIR, "preview");
const VERIFICATION_DIR = path.join(SCRATCH_DIR, "verification");
const INSPECT_PATH = path.join(SCRATCH_DIR, "inspect.ndjson");
const MAX_RENDER_VERIFY_LOOPS = 3;

const NAVY = "#0F172A";
const NAVY_SOFT = "#15233B";
const INK = "#111827";
const SLATE = "#334155";
const MUTED = "#64748B";
const PAPER = "#F8FAFC";
const PAPER_WARM = "#F8F6F1";
const CARD = "#FFFFFF";
const ACCENT = "#22C55E";
const ACCENT_DARK = "#166534";
const GOLD = "#D4A72C";
const CORAL = "#E76F51";
const SKY = "#60A5FA";
const BORDER = "#CBD5E1";
const TRANSPARENT = "#00000000";

const TITLE_FACE = "Aptos Display";
const BODY_FACE = "Aptos";
const MONO_FACE = "Aptos Mono";

const SOURCES = {
  registration: "photos_for_slides/court_reg_v3_summary_report.json",
  demoSummary: "photos_for_slides/demo_foot_fault_summary.json",
  demoEvents: "photos_for_slides/demo_foot_fault_events.csv",
  summary: "photos_for_slides/TECHNICAL_SUMMARY.md",
};

const IMAGES = {
  cover: path.join(REF_DIR, "01_registration_overlay.png"),
  comparison: path.join(REF_DIR, "02_registration_comparison.png"),
  fault: path.join(REF_DIR, "03_detected_fault_event.png"),
  uncertain: path.join(REF_DIR, "04_uncertain_event.png"),
  review: path.join(REF_DIR, "05_uncertain_review_event.png"),
};

const inspectRecords = [];

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function ensureDirs() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(SCRATCH_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(VERIFICATION_DIR, { recursive: true });
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function line(fill = BORDER, width = 1.2) {
  return { style: "solid", fill, width };
}

function record(kind, payload) {
  inspectRecords.push({ kind, ...payload });
}

function addShape(slide, slideNo, geometry, left, top, width, height, fill, outline = TRANSPARENT, outlineWidth = 0, role = geometry) {
  const shape = slide.shapes.add({
    geometry,
    position: { left, top, width, height },
    fill,
    line: line(outline, outlineWidth),
  });
  record("shape", { slide: slideNo, role, shapeType: geometry, bbox: [left, top, width, height], id: shape?.id });
  return shape;
}

function addText(
  slide,
  slideNo,
  text,
  left,
  top,
  width,
  height,
  {
    size = 20,
    color = INK,
    bold = false,
    typeface = BODY_FACE,
    align = "left",
    valign = "top",
    fill = TRANSPARENT,
    outline = TRANSPARENT,
    outlineWidth = 0,
    autoFit = "shrinkText",
    role = "text",
    insets = { left: 0, right: 0, top: 0, bottom: 0 },
  } = {},
) {
  const shape = addShape(slide, slideNo, "rect", left, top, width, height, fill, outline, outlineWidth, role);
  shape.text = text;
  shape.text.fontSize = size;
  shape.text.color = color;
  shape.text.bold = bold;
  shape.text.typeface = typeface;
  shape.text.alignment = align;
  shape.text.verticalAlignment = valign;
  shape.text.insets = insets;
  if (autoFit) {
    shape.text.autoFit = autoFit;
  }
  record("textbox", {
    slide: slideNo,
    role,
    text: String(text ?? ""),
    textChars: String(text ?? "").length,
    textLines: String(text ?? "").split(/\n/).length,
    bbox: [left, top, width, height],
    id: shape?.id,
  });
  return shape;
}

async function addImage(slide, slideNo, imagePath, left, top, width, height, { fit = "contain", geometry = "roundRect", role = "image", alt = "Slide image" } = {}) {
  const image = slide.images.add({
    blob: await readImageBlob(imagePath),
    fit,
    alt,
  });
  image.position = { left, top, width, height };
  image.geometry = geometry;
  record("image", { slide: slideNo, role, path: imagePath, bbox: [left, top, width, height], id: image?.id });
  return image;
}

function addHeader(slide, slideNo, label) {
  addText(slide, slideNo, label.toUpperCase(), 64, 32, 400, 22, {
    size: 12,
    bold: true,
    color: ACCENT_DARK,
    typeface: MONO_FACE,
    role: "header label",
    autoFit: null,
  });
  addText(slide, slideNo, `${String(slideNo).padStart(2, "0")} / 10`, 1120, 32, 96, 22, {
    size: 12,
    bold: true,
    color: ACCENT_DARK,
    typeface: MONO_FACE,
    role: "header index",
    align: "right",
    autoFit: null,
  });
  addShape(slide, slideNo, "rect", 64, 60, 1152, 2, BORDER, TRANSPARENT, 0, "header rule");
  addShape(slide, slideNo, "ellipse", 56, 52, 16, 16, ACCENT, TRANSPARENT, 0, "header marker");
}

function addMetricCard(slide, slideNo, left, top, width, height, metric, label, note, accent = ACCENT) {
  addShape(slide, slideNo, "roundRect", left, top, width, height, CARD, BORDER, 1.2, `metric card ${label}`);
  addShape(slide, slideNo, "rect", left, top, width, 6, accent, TRANSPARENT, 0, `metric accent ${label}`);
  addText(slide, slideNo, metric, left + 20, top + 18, width - 40, 46, {
    size: 30,
    bold: true,
    color: INK,
    typeface: TITLE_FACE,
    role: `metric ${label}`,
  });
  addText(slide, slideNo, label, left + 20, top + 72, width - 40, 42, {
    size: 15,
    color: SLATE,
    role: `metric label ${label}`,
  });
  if (note) {
    addText(slide, slideNo, note, left + 20, top + height - 36, width - 40, 18, {
      size: 10,
      color: MUTED,
      role: `metric note ${label}`,
      autoFit: null,
    });
  }
}

function addTakeawayCard(slide, slideNo, left, top, width, height, title, body, accent = ACCENT) {
  addShape(slide, slideNo, "roundRect", left, top, width, height, CARD, BORDER, 1.2, `takeaway ${title}`);
  addShape(slide, slideNo, "rect", left, top, 8, height, accent, TRANSPARENT, 0, `takeaway accent ${title}`);
  addText(slide, slideNo, title, left + 26, top + 18, width - 46, 26, {
    size: 15,
    bold: true,
    color: ACCENT_DARK,
    typeface: MONO_FACE,
    role: `takeaway title ${title}`,
    autoFit: null,
  });
  addText(slide, slideNo, body, left + 26, top + 54, width - 46, height - 70, {
    size: 17,
    color: INK,
    role: `takeaway body ${title}`,
  });
}

function addBulletListCard(slide, slideNo, left, top, width, height, title, items, accent = ACCENT) {
  addShape(slide, slideNo, "roundRect", left, top, width, height, CARD, BORDER, 1.2, `bullet card ${title}`);
  addShape(slide, slideNo, "rect", left, top, 8, height, accent, TRANSPARENT, 0, `bullet accent ${title}`);
  addText(slide, slideNo, title, left + 24, top + 18, width - 40, 26, {
    size: 15,
    bold: true,
    color: ACCENT_DARK,
    typeface: MONO_FACE,
    role: `bullet title ${title}`,
    autoFit: null,
  });
  const body = items.map((item) => `• ${item}`).join("\n");
  addText(slide, slideNo, body, left + 24, top + 54, width - 42, height - 72, {
    size: 17,
    color: INK,
    role: `bullet body ${title}`,
  });
}

function addStageCard(slide, slideNo, left, top, width, height, step, title, body, accent = ACCENT) {
  addShape(slide, slideNo, "roundRect", left, top, width, height, CARD, BORDER, 1.2, `stage ${title}`);
  addShape(slide, slideNo, "ellipse", left + 18, top + 18, 38, 38, accent, TRANSPARENT, 0, `stage badge ${title}`);
  addText(slide, slideNo, step, left + 27, top + 25, 20, 12, {
    size: 14,
    bold: true,
    color: CARD,
    typeface: MONO_FACE,
    align: "center",
    valign: "middle",
    role: `stage number ${title}`,
    autoFit: null,
  });
  addText(slide, slideNo, title, left + 70, top + 18, width - 86, 24, {
    size: 17,
    bold: true,
    color: INK,
    role: `stage title ${title}`,
    autoFit: null,
  });
  addText(slide, slideNo, body, left + 24, top + 66, width - 48, height - 84, {
    size: 15,
    color: SLATE,
    role: `stage body ${title}`,
  });
}

async function addPhotoPanel(slide, slideNo, imagePath, left, top, width, height, role, fit = "contain") {
  addShape(slide, slideNo, "roundRect", left, top, width, height, CARD, BORDER, 1.2, `${role} frame`);
  await addImage(slide, slideNo, imagePath, left + 10, top + 10, width - 20, height - 20, {
    fit,
    geometry: "roundRect",
    role,
    alt: role,
  });
}

function addNotes(slide, body, sourceKeys) {
  const sourceText = sourceKeys.map((key) => `- ${SOURCES[key] || key}`).join("\n");
  slide.speakerNotes.setText(`${body}\n\n[Sources]\n${sourceText}`);
}

async function slide1(presentation) {
  const slideNo = 1;
  const slide = presentation.slides.add();
  slide.background.fill = NAVY;
  addShape(slide, slideNo, "roundRect", 662, 72, 558, 578, "#1E293BCC", TRANSPARENT, 0, "hero glow");
  await addPhotoPanel(slide, slideNo, IMAGES.cover, 678, 88, 526, 484, "registration overlay", "contain");
  addText(slide, slideNo, "KITCHENMASTER", 72, 92, 250, 20, {
    size: 12,
    bold: true,
    color: "#93C5FD",
    typeface: MONO_FACE,
    role: "cover kicker",
    autoFit: null,
  });
  addText(slide, slideNo, "Pickleball kitchen foot-fault detection from a single side-view camera", 72, 126, 560, 170, {
    size: 34,
    bold: true,
    color: PAPER,
    typeface: TITLE_FACE,
    role: "cover title",
  });
  addText(slide, slideNo, "A research prototype that registers court geometry, infers likely volley events, localizes the relevant foot, and labels the result as legal, fault, or uncertain.", 74, 310, 540, 96, {
    size: 20,
    color: "#DCE7F5",
    role: "cover subtitle",
  });
  addMetricCard(slide, slideNo, 72, 454, 166, 118, "2055 / 2055", "registered frames", "best current technical result", ACCENT);
  addMetricCard(slide, slideNo, 252, 454, 166, 118, "1 fault", "demo snapshot", "plus 2 uncertain events", GOLD);
  addMetricCard(slide, slideNo, 432, 454, 166, 118, "1 camera", "capture setup", "single fixed side view", SKY);
  addText(slide, slideNo, "Main claim: geometry is the foundation, and uncertainty is better than a wrong fault call.", 74, 606, 544, 54, {
    size: 18,
    color: "#DCE7F5",
    role: "cover claim",
  });
  addNotes(
    slide,
    "Open with the problem framing, then immediately anchor the audience on the strongest result: reliable court registration across the full clip.",
    ["summary", "registration", "demoSummary"],
  );
}

async function slide2(presentation) {
  const slideNo = 2;
  const slide = presentation.slides.add();
  slide.background.fill = PAPER;
  addHeader(slide, slideNo, "Problem");
  addText(slide, slideNo, "Why this problem is hard from a single camera", 64, 88, 560, 82, {
    size: 31,
    bold: true,
    color: INK,
    typeface: TITLE_FACE,
    role: "title",
  });
  addText(slide, slideNo, "A foot location alone is not enough. The system has to know where the kitchen line is in the same frame, which player is active, and whether the evidence is clear enough to support a confident call.", 64, 176, 548, 80, {
    size: 18,
    color: SLATE,
    role: "subtitle",
  });
  addBulletListCard(slide, slideNo, 64, 280, 530, 164, "Research questions", [
    "Can a side-view camera detect whether a foot stayed behind the kitchen line or crossed it?",
    "How sensitive is the decision to blur, occlusion, angle, and line ambiguity?",
    "Is it better to output uncertain than force a wrong legal or fault label?",
  ], ACCENT);
  addBulletListCard(slide, slideNo, 64, 466, 530, 156, "What the project must estimate", [
    "Court geometry in each frame",
    "Likely volley timing from the ball trajectory",
    "Relevant foot point and signed distance to the NVZ boundary",
  ], GOLD);
  await addPhotoPanel(slide, slideNo, IMAGES.cover, 642, 130, 572, 436, "problem framing image", "contain");
  addTakeawayCard(
    slide,
    slideNo,
    642,
    584,
    572,
    98,
    "Key takeaway",
    "Geometry is the foundation. Without a stable kitchen-line estimate, every later label becomes hard to trust.",
    CORAL,
  );
  addNotes(
    slide,
    "Use this slide to establish the problem before getting into methods. The line estimate, active side, and foot point all have to agree in the same frame.",
    ["summary"],
  );
}

async function slide3(presentation) {
  const slideNo = 3;
  const slide = presentation.slides.add();
  slide.background.fill = PAPER_WARM;
  addHeader(slide, slideNo, "Pipeline");
  addText(slide, slideNo, "Pipeline overview: from video to decision", 64, 88, 560, 82, {
    size: 31,
    bold: true,
    color: INK,
    typeface: TITLE_FACE,
    role: "title",
  });
  addText(slide, slideNo, "The current system is a configurable pipeline: register the court, track the ball, infer events, localize the relevant foot, then apply a signed-distance rule with an uncertainty margin.", 64, 174, 720, 70, {
    size: 18,
    color: SLATE,
    role: "subtitle",
  });
  addStageCard(slide, slideNo, 64, 272, 350, 124, "1", "Court registration", "Estimate where the kitchen boundaries are in each frame using anchor points and ORB-based stabilization.", ACCENT);
  addStageCard(slide, slideNo, 442, 272, 350, 124, "2", "Ball tracking", "Track the ball trajectory and export candidate positions plus debug overlays.", GOLD);
  addStageCard(slide, slideNo, 820, 272, 350, 124, "3", "Event inference", "Convert the trajectory into bounce or volley timing hypotheses for likely review frames.", CORAL);
  addStageCard(slide, slideNo, 160, 432, 350, 124, "4", "Foot localization", "Estimate the relevant contact point using the event_hybrid foot localizer and ROI constraints.", SKY);
  addStageCard(slide, slideNo, 542, 432, 350, 124, "5", "Signed-distance label", "Measure the foot against the selected NVZ boundary and output legal, fault, or uncertain.", ACCENT);
  addStageCard(slide, slideNo, 924, 432, 194, 124, "6", "Review", "Stop for human correction on hard cases.", GOLD);
  addTakeawayCard(slide, slideNo, 64, 584, 1106, 86, "Interpretability", "The final label is not a black box. It can be traced back to court geometry, player side, foot point, and a measurable signed distance.", ACCENT);
  addNotes(
    slide,
    "This slide is the method summary. Keep it high level and save implementation details for questions.",
    ["summary"],
  );
}

async function slide4(presentation) {
  const slideNo = 4;
  const slide = presentation.slides.add();
  slide.background.fill = PAPER_WARM;
  addHeader(slide, slideNo, "Registration Foundation");
  addText(slide, slideNo, "Court registration is the strongest result in the repo", 64, 88, 550, 86, {
    size: 31,
    bold: true,
    color: INK,
    typeface: TITLE_FACE,
    role: "title",
  });
  addText(slide, slideNo, "The anchor-point court model plus ORB post-translation tracked the court across the full 34.28 second clip with no fallbacks.", 64, 176, 536, 60, {
    size: 18,
    color: SLATE,
    role: "subtitle",
  });
  addMetricCard(slide, slideNo, 64, 268, 170, 120, "2055 / 2055", "successful frames", "0.0% fallback rate", ACCENT);
  addMetricCard(slide, slideNo, 248, 268, 170, 120, "0", "fallback frames", "reference clip at 1920x1080", GOLD);
  addMetricCard(slide, slideNo, 432, 268, 170, 120, "25.03 px", "mean translation", "sampled validation frames", SKY);
  await addPhotoPanel(slide, slideNo, IMAGES.cover, 642, 126, 572, 430, "registration overlay", "contain");
  addTakeawayCard(
    slide,
    slideNo,
    64,
    412,
    536,
    204,
    "Why this matters",
    "Everything downstream depends on knowing where the kitchen line actually is in each frame. The demo is only interpretable because the boundary is registered before any fault decision is made.",
    ACCENT,
  );
  addText(slide, slideNo, "The overlay image makes the project easier to explain: once the court is aligned, later foot-fault measurements become a geometric question instead of a guess.", 642, 574, 566, 58, {
    size: 16,
    color: MUTED,
    role: "overlay caption",
  });
  addNotes(
    slide,
    "Emphasize that registration is the most defensible result. If asked for the single strongest number, cite 2055 out of 2055 frames with zero fallbacks.",
    ["registration", "summary"],
  );
}

async function slide5(presentation) {
  const slideNo = 5;
  const slide = presentation.slides.add();
  slide.background.fill = PAPER;
  addHeader(slide, slideNo, "Comparison");
  addText(slide, slideNo, "The current registration method is more stable than the affine comparison", 64, 88, 560, 112, {
    size: 31,
    bold: true,
    color: INK,
    typeface: TITLE_FACE,
    role: "title",
  });
  addText(slide, slideNo, "On the same clip, the fixed affine configuration dropped from 2055 successful frames to 2036 and needed 19 fallbacks.", 64, 198, 540, 58, {
    size: 18,
    color: SLATE,
    role: "subtitle",
  });
  addMetricCard(slide, slideNo, 64, 272, 170, 118, "2055", "post-translation", "successful frames", ACCENT);
  addMetricCard(slide, slideNo, 248, 272, 170, 118, "2036", "affine fixed", "successful frames", CORAL);
  addMetricCard(slide, slideNo, 432, 272, 170, 118, "19", "affine fallbacks", "same clip and settings family", GOLD);
  addTakeawayCard(slide, slideNo, 64, 424, 536, 176, "Interpretation", "The registration stage is already giving a meaningful technical result: the current transform choice matters, and the more stable option is visible both qualitatively and in the fallback count.", ACCENT);
  await addPhotoPanel(slide, slideNo, IMAGES.comparison, 642, 128, 572, 430, "registration comparison", "contain");
  addText(slide, slideNo, "Left: post-translation. Right: affine. Even without going deep into the math, the qualitative comparison supports the same story as the frame-count metrics.", 644, 576, 566, 60, {
    size: 16,
    color: MUTED,
    role: "comparison caption",
  });
  addNotes(
    slide,
    "Use this slide to justify why the current registration configuration is not arbitrary. The comparison result is a strong way to show engineering progress.",
    ["registration", "summary"],
  );
}

async function slide6(presentation) {
  const slideNo = 6;
  const slide = presentation.slides.add();
  slide.background.fill = PAPER_WARM;
  addHeader(slide, slideNo, "Demo Snapshot");
  addText(slide, slideNo, "Current end-to-end demo status", 64, 88, 520, 76, {
    size: 31,
    bold: true,
    color: INK,
    typeface: TITLE_FACE,
    role: "title",
  });
  addText(slide, slideNo, "The demo pipeline is wired and producing reviewable outputs, but it should still be presented as a human-in-the-loop research workflow rather than a finished automatic referee.", 64, 170, 630, 70, {
    size: 18,
    color: SLATE,
    role: "subtitle",
  });
  addMetricCard(slide, slideNo, 64, 270, 170, 118, "3", "final events", "demo snapshot", ACCENT);
  addMetricCard(slide, slideNo, 248, 270, 170, 118, "1", "fault", "foot_fault_volley", GOLD);
  addMetricCard(slide, slideNo, 432, 270, 170, 118, "2", "uncertain", "hard cases stay reviewable", CORAL);
  addMetricCard(slide, slideNo, 616, 270, 170, 118, "35.7%", "ball detection rate", "weakest current stage", SKY);
  addBulletListCard(slide, slideNo, 64, 428, 542, 186, "What is already working", [
    "Registration outputs are stable and explainable.",
    "The pipeline produces event-level artifacts and summaries.",
    "At least one clean event is labeled as a foot fault end-to-end.",
  ], ACCENT);
  addBulletListCard(slide, slideNo, 636, 428, 542, 186, "What is still fragile", [
    "Ball detection and event timing are the weakest automatic stages.",
    "Some hard cases still require manual review and overrides.",
    "This is a demo workflow, not yet a production-grade referee.",
  ], CORAL);
  addNotes(
    slide,
    "This is the status slide. It is intentionally honest and should lower the risk of overclaiming the current system.",
    ["demoSummary", "summary"],
  );
}

async function slide7(presentation) {
  const slideNo = 7;
  const slide = presentation.slides.add();
  slide.background.fill = PAPER;
  addHeader(slide, slideNo, "Detected Fault");
  addText(slide, slideNo, "The end-to-end demo can surface a clear foot-fault example", 64, 88, 560, 82, {
    size: 31,
    bold: true,
    color: INK,
    typeface: TITLE_FACE,
    role: "title",
  });
  addText(slide, slideNo, "This event is labeled `foot_fault_volley` because the selected foot lands inside the non-volley zone after registration and active-side selection.", 64, 174, 540, 66, {
    size: 18,
    color: SLATE,
    role: "subtitle",
  });
  await addPhotoPanel(slide, slideNo, IMAGES.fault, 64, 258, 700, 378, "fault event", "contain");
  addMetricCard(slide, slideNo, 810, 262, 182, 118, "3", "events in demo", "presentation snapshot", ACCENT);
  addMetricCard(slide, slideNo, 1006, 262, 182, 118, "1", "fault event", "two more marked uncertain", GOLD);
  addMetricCard(slide, slideNo, 810, 396, 182, 118, "-10.78 px", "signed distance", "fault side of threshold", CORAL);
  addMetricCard(slide, slideNo, 1006, 396, 182, 118, "5 px", "fault threshold", "margin before uncertainty", SKY);
  addTakeawayCard(
    slide,
    slideNo,
    810,
    522,
    378,
    114,
    "Interpretability",
    "The label is explainable: court line, active player side, localized foot point, and signed distance all remain visible in the frame.",
    ACCENT,
  );
  addNotes(
    slide,
    "Use this slide to show that the prototype is more than a registration demo. The full chain can produce an interpretable fault call on at least one clean event.",
    ["demoSummary", "demoEvents", "summary"],
  );
}

async function slide8(presentation) {
  const slideNo = 8;
  const slide = presentation.slides.add();
  slide.background.fill = PAPER_WARM;
  addHeader(slide, slideNo, "Uncertainty");
  addText(slide, slideNo, "Uncertain is a deliberate outcome, not a failure state", 64, 88, 540, 82, {
    size: 31,
    bold: true,
    color: INK,
    typeface: TITLE_FACE,
    role: "title",
  });
  addText(slide, slideNo, "When the foot is near the line, the ball timing is noisy, or the active side is less certain, abstaining is safer than forcing a wrong call.", 64, 174, 536, 74, {
    size: 18,
    color: SLATE,
    role: "subtitle",
  });
  addMetricCard(slide, slideNo, 64, 280, 170, 118, "2", "uncertain events", "out of 3 demo events", ACCENT);
  addMetricCard(slide, slideNo, 248, 280, 170, 118, "15 px", "uncertain margin", "configured guard band", GOLD);
  addMetricCard(slide, slideNo, 432, 280, 170, 118, "0.551", "active-side confidence", "example uncertain case", CORAL);
  addTakeawayCard(
    slide,
    slideNo,
    64,
    430,
    536,
    188,
    "Presentation framing",
    "The honest story is that registration is strong, while the ball and event stage remains fragile. The system already knows when it should defer to human review instead of pretending to be fully automatic.",
    ACCENT,
  );
  await addPhotoPanel(slide, slideNo, IMAGES.uncertain, 640, 140, 576, 468, "uncertain event", "contain");
  addText(slide, slideNo, "This example stays close enough to the line that the safer label is `uncertain`, especially given the active-side confidence and ball ambiguity in the same frame.", 642, 622, 572, 50, {
    size: 15,
    color: MUTED,
    role: "uncertain caption",
  });
  addNotes(
    slide,
    "If asked why uncertainty is useful, answer that a wrong fault call is worse than abstaining. This slide is the evidence for that design choice.",
    ["demoSummary", "demoEvents", "summary"],
  );
}

async function slide9(presentation) {
  const slideNo = 9;
  const slide = presentation.slides.add();
  slide.background.fill = NAVY_SOFT;
  addHeader(slide, slideNo, "Human Review");
  addText(slide, slideNo, "The demo is intentionally human-in-the-loop", 64, 88, 548, 80, {
    size: 31,
    bold: true,
    color: PAPER,
    typeface: TITLE_FACE,
    role: "title",
  });
  addText(slide, slideNo, "In auto_review mode, the pipeline exports artifacts and stops so the reviewer can correct ambiguous frames before final outputs are generated.", 64, 172, 552, 72, {
    size: 18,
    color: "#DCE7F5",
    role: "subtitle",
  });
  await addPhotoPanel(slide, slideNo, IMAGES.review, 64, 264, 620, 340, "review event", "contain");
  addBulletListCard(slide, slideNo, 724, 264, 492, 160, "Reviewer can override", [
    "Line geometry",
    "Ball points and bounce labels",
    "Foot points and active side",
    "Final event label",
  ], ACCENT);
  addTakeawayCard(slide, slideNo, 724, 446, 492, 158, "Why this is acceptable in a research prototype", "The review step makes the workflow honest and auditable. Hard cases are surfaced explicitly instead of being hidden behind a false sense of automation.", GOLD);
  addText(slide, slideNo, "This architecture is a strength for presentation: it shows the system knows when to ask for help.", 64, 628, 1152, 36, {
    size: 15,
    color: "#DCE7F5",
    role: "review caption",
  });
  addNotes(
    slide,
    "This is the slide to use when the professor asks whether the system is fully automatic. The correct answer is no, and the review workflow is an intentional part of the current design.",
    ["summary"],
  );
}

async function slide10(presentation) {
  const slideNo = 10;
  const slide = presentation.slides.add();
  slide.background.fill = NAVY_SOFT;
  addHeader(slide, slideNo, "Takeaways");
  addText(slide, slideNo, "Present this as a research prototype with human review", 64, 88, 550, 86, {
    size: 31,
    bold: true,
    color: PAPER,
    typeface: TITLE_FACE,
    role: "title",
  });
  addText(slide, slideNo, "That framing highlights what already works, stays honest about what does not, and gives a credible roadmap for future improvement.", 64, 176, 544, 62, {
    size: 18,
    color: "#DCE7F5",
    role: "subtitle",
  });
  await addPhotoPanel(slide, slideNo, IMAGES.review, 64, 264, 640, 356, "review event", "contain");
  addTakeawayCard(slide, slideNo, 744, 262, 470, 106, "Strongest result", "Court registration stayed stable across 2055 / 2055 frames with 0 fallbacks on the real clip.", ACCENT);
  addTakeawayCard(slide, slideNo, 744, 388, 470, 106, "Current limitation", "Ball tracking and event timing still limit the automatic pipeline, which is why the demo uses a review-and-override workflow.", GOLD);
  addTakeawayCard(slide, slideNo, 744, 514, 470, 106, "Best next step", "Improve the ball detector and collect more labeled real clips so the uncertainty and override rates can be measured rigorously.", CORAL);
  addText(slide, slideNo, "Recommended closing line: KitchenMaster already produces interpretable court-aligned event evidence, but it should be presented today as a human-in-the-loop referee assistant rather than a finished autonomous judge.", 64, 638, 1150, 38, {
    size: 15,
    color: "#DCE7F5",
    role: "closing line",
  });
  addNotes(
    slide,
    "Finish with the strongest result, the honest limitation, and the next step. This keeps the presentation credible and focused.",
    ["summary", "registration", "demoSummary"],
  );
}

async function createDeck() {
  await ensureDirs();
  for (const imagePath of Object.values(IMAGES)) {
    if (!(await pathExists(imagePath))) {
      throw new Error(`Missing required image: ${imagePath}`);
    }
  }
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  await slide1(presentation);
  await slide2(presentation);
  await slide3(presentation);
  await slide4(presentation);
  await slide5(presentation);
  await slide6(presentation);
  await slide7(presentation);
  await slide8(presentation);
  await slide9(presentation);
  await slide10(presentation);
  return presentation;
}

async function saveBlobToFile(blob, filePath) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  await fs.writeFile(filePath, bytes);
}

async function writeInspectArtifact(presentation) {
  const lines = [
    JSON.stringify({ kind: "deck", id: DECK_ID, slideCount: presentation.slides.count, slideSize: { width: W, height: H } }),
    ...presentation.slides.items.map((slide, index) => JSON.stringify({ kind: "slide", slide: index + 1, id: slide?.id || `slide-${index + 1}` })),
    ...inspectRecords.map((recordItem) => JSON.stringify(recordItem)),
  ];
  await fs.writeFile(INSPECT_PATH, lines.join("\n") + "\n", "utf8");
}

async function currentRenderLoopCount() {
  const logPath = path.join(VERIFICATION_DIR, "render_verify_loops.ndjson");
  if (!(await pathExists(logPath))) return 0;
  const content = await fs.readFile(logPath, "utf8");
  return content.split(/\r?\n/).filter(Boolean).length;
}

async function appendRenderVerifyLoop(presentation, previewPaths, pptxPath) {
  const logPath = path.join(VERIFICATION_DIR, "render_verify_loops.ndjson");
  const loop = (await currentRenderLoopCount()) + 1;
  const recordItem = {
    kind: "render_verify_loop",
    deckId: DECK_ID,
    loop,
    maxLoops: MAX_RENDER_VERIFY_LOOPS,
    previewCount: previewPaths.length,
    pptxPath,
    timestamp: new Date().toISOString(),
  };
  await fs.appendFile(logPath, JSON.stringify(recordItem) + "\n", "utf8");
  return recordItem;
}

async function verifyAndExport(presentation) {
  await ensureDirs();
  const nextLoop = (await currentRenderLoopCount()) + 1;
  if (nextLoop > MAX_RENDER_VERIFY_LOOPS) {
    throw new Error(`Render loop cap reached: ${MAX_RENDER_VERIFY_LOOPS}`);
  }
  await writeInspectArtifact(presentation);
  const previewPaths = [];
  for (let idx = 0; idx < presentation.slides.items.length; idx += 1) {
    const slide = presentation.slides.items[idx];
    const preview = await presentation.export({ slide, format: "png", scale: 1 });
    const previewPath = path.join(PREVIEW_DIR, `slide-${String(idx + 1).padStart(2, "0")}.png`);
    await saveBlobToFile(preview, previewPath);
    previewPaths.push(previewPath);
  }
  const pptxPath = path.join(OUT_DIR, "output.pptx");
  const pptxBlob = await PresentationFile.exportPptx(presentation);
  await pptxBlob.save(pptxPath);
  await appendRenderVerifyLoop(presentation, previewPaths, pptxPath);
  return { pptxPath, previewPaths };
}

const presentation = await createDeck();
const result = await verifyAndExport(presentation);
console.log(result.pptxPath);
