# KitchenMaster Final Report Bundle

This folder is a self-contained report-prep bundle for the IEEE-style final report. It gathers the repo's strongest narrative, code, configs, tests, numeric outputs, slide-ready assets, and generated summary tables in one place.

## Read These First

1. `01_FINAL_REPORT_BRIEF.md` - concise report-ready story, methods, results, limitations, and figure suggestions.
2. `02_RESULTS_AND_FIGURES.md` - exact metrics, copied assets, generated charts, and missing artifact warnings.
3. `03_METHODS_CODE_INDEX.md` - implementation map with the files/functions that matter.
4. `04_REPRODUCIBILITY_AND_GAPS.md` - commands, test status, missing outputs, and what to rerun if you need a cleaner final submission.
5. `CLAUDE_PASTE_CONTEXT.md` - compact one-file context intended for pasting into Claude or another model.

## Folder Map

- `assets/` - copied PNG/JPG/MP4/PPTX assets useful for figures, appendix, or visual evidence.
- `data_results/` - copied JSON/CSV experiment outputs, annotations, review files, and prior export bundles.
- `generated_charts/` - lightweight SVG charts generated from existing results.
- `source_snapshot/` - relevant source, experiment runners, configs, scripts, and unit tests.
- `docs/` and `prior_context/` - narrative files, prior summaries, and prompt/chat context.
- `tables/` - generated CSV summary tables.
- `bundle_manifest.csv` - every copied/generated file with source path, size, and SHA-256.
- `missing_or_regenerate.csv` - files referenced by reports/configs but absent in the current repo snapshot.

## Best Current Result

Court registration v3 registered `2055/2055` frames with `0` fallbacks on `pickle_vid_1_trimmed_from_8s.mp4`. The comparison report shows `post_translation` had `0` fallback frames while the affine comparison had `19`.
