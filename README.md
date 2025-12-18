# Semi-Automated Central Review for Glioblastoma (RANO / PSP)

Research prototype for **semi-automated central review of glioblastoma MRI**, designed to support longitudinal RANO assessment and **pseudoprogression (PSP) risk stratification** using quantitative imaging-derived features.

This repository contains:
- a **stable, tabular-only clinical demo (v1)**  
- an **experimental multimodal research extension (v2)**  

The system is intended **for research and methodological exploration only**.

---

## Dataset

**LUMIERE — Longitudinal Glioblastoma MRI with Expert RANO Evaluation**  
Source: Figshare (non-commercial use only)

Key dataset characteristics:
- longitudinal MRI follow-up
- week-normalized timeline from pre-operative baseline
- automatic tumor segmentations  
  (DeepBraTumIA, HD-GLIO-AUTO + HD-BET skull-stripping)

> This repository **does not perform segmentation**.  
> It operates exclusively on pre-computed segmentations and derived features.

---

## Versioning Overview

| Version | Scope | Status | Intended Use |
|------|------|------|------|
| **v1 — Tabular-Only** | Volumetric + longitudinal features | **Stable** | Clinical demo |
| **v2 — Multimodal** | Tabular + CLIP embeddings | Experimental | Research only |

The **clinical demo runs exclusively on v1**.  
v2 is preserved to document additional research work and future directions.

---

# v1 — Tabular-Only Clinical Demo (Stable)

## What the Demo Does

The v1 pipeline performs **PSP risk stratification** using **tabular longitudinal features only**.

At inference time, the model consumes:
- enhancing / non-enhancing / edema volumes
- longitudinal deltas and ratios
- baseline-normalized progression indicators
- week-normalized temporal features

MRI images are used **only for visualization**, not as direct model inputs.

---

## v1 Pipeline

```

Tabular longitudinal features
→ Calibrated Logistic Regression
→ LightGBM
→ Stacked meta-model
→ Isotonic probability calibration
→ RED / YELLOW / GREEN triage

```

---

## v1 Golden path

```

python models/export_stack_v1.py \
  --sot source_of_truth/v1_tabular_only \
  --from_parquet --patient Patient-019 --week_norm 15

```

```
python models/export_stack_v1.py \
  --sot source_of_truth/v1_tabular_only \
  --from_parquet --patient Patient-019 --week_norm 15 \
  --out report/demo_outputs/Patient-019_w015.json
```

---


## Repository Structure (v1)

```

central-review-rano-gbm/
├── source_of_truth/
│   └── v1_tabular_only/
│       ├── models/                 # Trained models + calibration
│       ├── thresholds/             # Active triage thresholds
│       ├── demo_inputs/            # Sample JSON inputs
│       └── README.md
├── scripts/
├── models/export_stack_v1.py       # Inference entrypoint
├── data/processed/
└── report/

````

---

## Training & Evaluation (v1)

Leave-One-Patient-Out Cross-Validation (LOOCV) is used throughout.

```bash
python scripts/train_psp_multimodal_stack.py \
  --data data/processed/psp_multimodal.parquet \
  --mode tabular_only \
  --outdir report/psp
````

Outputs:

* calibrated LR, LGBM, and stacked models
* OOF predictions
* bootstrap confidence intervals
* Decision Curve Analysis (DCA)
* triage thresholds

---

## Triage Scheme

Thresholds are derived via **Decision Curve Analysis**.

Example:

```
RED     ≥ 0.61
YELLOW  ≥ 0.33
GREEN   < 0.33
```

Thresholds are versioned and stored explicitly.

---

## Inference — Recommended Mode (Automatic)

Single-command inference from the processed dataset:

```bash
python models/export_stack_v1.py \
  --from_parquet \
  --patient Patient-019 \
  --week_norm 15
```

The script will:

1. Load tabular features for the specified patient/timepoint
2. Assemble the feature vector in **exact training order**
3. Run LR, LGBM, and stacked prediction
4. Apply isotonic calibration
5. Output PSP probability and triage class

Example output:

```json
{
  "p_lr": 0.46,
  "p_lgbm": 0.62,
  "p_stack": 0.67,
  "p_stack_iso": 0.88,
  "triage": "RED"
}
```

---

## Inference — Manual Mode (Testing Only)

```bash
python models/export_stack_v1.py \
  --json_features source_of_truth/v1_tabular_only/demo_inputs/sample_features.json
```

Use for controlled experiments only.

---

# v2 — Multimodal Research Extension (Experimental)

## Overview

The v2 pipeline extends v1 by integrating **imaging embeddings** extracted from standardized MRI montages.

Additional components:

* CLIP embeddings (ViT-H/14, LAION2B)
* 2×2 MRI montage input (T1, T1CE, T2, FLAIR)
* PCA dimensionality reduction
* Multimodal stacking

This pipeline is **not part of the clinical demo**.

---

## v2 Pipeline

```
MRI montage
→ CLIP embedding (ViT-H/14)
→ L2 normalization
→ PCA compression
→ Multimodal feature join
→ LR + LGBM
→ Stacked prediction
→ Calibration & triage
```

---

## Research Notes

* PCA is fit **only on training embeddings**
* Montage layout must remain consistent
* Embeddings are sensitive to acquisition protocol shifts
* v2 is preserved for:

  * ablation studies
  * representation learning experiments
  * future multimodal extensions

---

# Clinical & Safety Disclaimers

## Intended Use

**Research use only.**
This system is **not** a clinical decision-support tool and is **not validated** for patient care.

It is intended for:

* algorithmic experimentation
* model calibration research
* reproducible imaging-based analyses

It is **not intended** for:

* clinical diagnosis
* treatment selection
* therapy response assessment
* deployment in clinical workflows

---

## Labeling Disclaimer

PSP labels used in this repository are derived from **AI ensemble consensus**, not certified radiologist ground truth.

They represent **imaging phenotypes**, not clinical diagnoses.

---

## Safety Notice

All outputs — probabilities, triage classes, risk scores — must be treated as **experimental predictions**.

No clinical, diagnostic, or therapeutic decisions should be based on this system.

---

## Design Principles

* Explicit versioning (models, thresholds, feature order)
* No hidden state
* Reproducible pipelines
* Fast per-timepoint inference
* Research-grade calibration over raw accuracy

---

## Contact & Attribution

This repository builds on:

* LUMIERE dataset (Scientific Data, 2022)
* Prior longitudinal GBM feature extraction work

For questions or collaboration, please open an issue or contact the maintainers.
