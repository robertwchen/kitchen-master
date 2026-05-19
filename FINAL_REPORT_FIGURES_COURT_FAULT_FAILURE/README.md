# Final Report Figures: Court, Uncertain Event, Comparison

This folder contains actual KitchenMaster pipeline outputs selected for the final report.

1. `Figure_1_Court_Registration_Overlay.png`
   - Source: `FINAL_REPORT_HANDOFF_COMPLETE/figures/photos_for_slides/01_registration_overlay.png`
   - Use as Figure 1: strongest court registration overlay. Shows registered NVZ/kitchen geometry aligned to real video.

2. `Figure_2_Uncertain_Event_Frame_1948.png`
   - Source: `FINAL_REPORT_HANDOFF_COMPLETE/figures/photos_for_slides/04_uncertain_event.png`
   - Use as Figure 2 if you want an honest event figure. Shows an actual uncertain event where active-side/foot-line geometry is ambiguous.

3. `Optional_Figure_3_Registration_Rolling_vs_Fixed_Frame_02000.png`
   - Source: `results/real_baseline/court_reg_v3/comparisons/rolling_vs_fixed/frame_02000.png`
   - Optional Figure 3: side-by-side registration comparison. Use this if you want to show the weaker rolling-reference registration on the left and the stronger fixed-reference result on the right.

Also available but not currently selected:
- `FINAL_REPORT_HANDOFF_COMPLETE/figures/photos_for_slides/05_uncertain_review_event.png` for a review-required uncertain case at frame 1537.
- `FINAL_REPORT_HANDOFF_COMPLETE/misc_analysis/ball_test/debug_all_candidates_f5.png` for ball-candidate/glare clutter limitations.

Do not use the old `03_detected_fault_event.png` unless you manually verify that its displayed signed-distance/label text matches the paper. The saved active-side event CSV reports frame 929 signed distance as -10.78 px, not -440 px.
