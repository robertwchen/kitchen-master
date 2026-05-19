Audience: professor / class presentation for a research prototype demo.

Objective: explain the problem, show how the pipeline works, highlight the strongest technical result, show end-to-end examples, and present the project honestly as a human-in-the-loop research prototype.

Narrative arc:
1. Introduce the project and the core problem.
2. Explain why geometry matters before classification.
3. Walk through the pipeline stages at a high level.
4. Show the strongest result in court registration.
5. Compare the current registration method against a weaker alternative.
6. Summarize the current end-to-end demo status.
7. Show a detected fault event.
8. Show why uncertain is an intentional output.
9. Explain the review workflow and why the system is not yet fully automatic.
10. Close with takeaways and next steps.

Slide list:
1. Cover / project framing with registration overlay image.
2. Problem framing and research questions.
3. Pipeline overview from video to label.
4. Registration foundation slide with overlay image and metrics.
5. Registration comparison slide with side-by-side comparison image.
6. Demo pipeline snapshot and current status metrics.
7. Detected fault event slide with event screenshot and decision metrics.
8. Uncertain event slide with uncertainty framing and thresholds.
9. Human review workflow slide with review screenshot and explanation.
10. Final takeaways and next steps.

Source plan:
- `photos_for_slides/01_registration_overlay.png`
- `photos_for_slides/02_registration_comparison.png`
- `photos_for_slides/03_detected_fault_event.png`
- `photos_for_slides/04_uncertain_event.png`
- `photos_for_slides/05_uncertain_review_event.png`
- `photos_for_slides/court_reg_v3_summary_report.json`
- `photos_for_slides/demo_foot_fault_summary.json`
- `photos_for_slides/demo_foot_fault_events.csv`
- `photos_for_slides/TECHNICAL_SUMMARY.md`

Visual system:
- 16:9 deck with dark navy + off-white palette and green accent.
- Large screenshot panels with rounded frames.
- Short editable headline text, metric cards, and takeaway cards.

Asset needs:
- No image generation required.
- Use the five curated screenshots in `photos_for_slides/`.

Editability plan:
- All titles, body text, metric values, and takeaway cards remain native editable text boxes.
- Screenshots are used only as placed images.
