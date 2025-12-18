# V2 Multimodal (Research Snapshot)

This folder is a **research snapshot** of the multimodal pipeline (tabular + CLIP PCA).
It is kept for provenance and to show the multimodal work, but the demo repo default is:

- **V1 Tabular-only**: runnable demo with minimal dependencies and no imaging assets.

What’s inside:
- `models/`: trained estimators + calibration artifacts
- `pca/` or `models/pca_clip.joblib`: PCA used for CLIP embeddings
- `thresholds/`: triage thresholds (if available)

Notes:
- Running V2 inference typically requires montage images + CLIP extraction dependencies (`torch`, `open_clip`, etc.)
- This repo does **not** ship raw clinical imaging data.
