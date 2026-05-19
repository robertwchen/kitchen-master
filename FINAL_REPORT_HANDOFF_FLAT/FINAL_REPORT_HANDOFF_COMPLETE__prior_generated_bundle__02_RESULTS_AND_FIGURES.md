# Results And Figures

## Copied Figure Assets

- `assets/photos_for_slides/01_registration_overlay.png` - primary registration overlay.
- `assets/photos_for_slides/02_registration_comparison.png` - registration comparison visual.
- `assets/photos_for_slides/03_detected_fault_event.png` - detected fault event.
- `assets/photos_for_slides/04_uncertain_event.png` - uncertain event.
- `assets/photos_for_slides/05_uncertain_review_event.png` - uncertainty/review example.
- `assets/photos_for_slides/registration_overlay_video.mp4` - dynamic registration evidence.
- `assets/outputs/slides/kitchenmaster-prof-deck/output.pptx` - presentation deck.
- `assets/tmp/slides/kitchenmaster-prof-deck/preview/*.png` - slide preview images.
- `assets/misc/ball_test/*.png` and `assets/misc/ball_test/debug_frames/*.png` - ball tracking debug evidence.
- `assets/misc/ball_analysis/**/*.jpg` - ball-analysis frame samples from night footage and brightness studies.

## Generated Charts

- `generated_charts/registration_fallback_comparison.svg` - fallback comparison for post-translation, affine, and static settings.
- `generated_charts/synthetic_rates.svg` - synthetic false-fault, missed-fault, and uncertainty rates.

## Generated Tables

- `tables/results_summary.csv` - consolidated metrics from JSON/CSV results.
- `tables/synthetic_confusion_matrix.csv` - confusion matrix generated from `predictions_full.csv`.

## Demo Event Rows

The copied `data_results/photos_for_slides/demo_foot_fault_events.csv` contains three events:

- Frame 929 at 15.4981 s: side=right, label=foot_fault_volley, signed_dist_px=-10.78, foot=(1574.33, 785.51), review_required=False.
- Frame 1537 at 25.641 s: side=right, label=uncertain, signed_dist_px=-41.93, foot=(1664.0, 800.0), review_required=True.
- Frame 1948 at 32.4975 s: side=left, label=uncertain, signed_dist_px=21.72, foot=(244.49, 779.54), review_required=True.


## Caution

Several JSON/CSV files reference event-frame PNGs, per-frame registration CSVs, and overlay videos that are not present in the current repo snapshot. See `missing_or_regenerate.csv` and `04_REPRODUCIBILITY_AND_GAPS.md` before citing those referenced paths as available artifacts.
