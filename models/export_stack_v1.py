#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings
from PIL import Image

# =========================
# Utils
# =========================

'''
def clinical_line(triage: str, p: float) -> str:
    """
    One-line clinical-style explanation for the demo.
    p is the probability used for triage (raw or calibrated).
    """
    triage = (triage or "").upper()

    for k in ("RED","YELLOW","GREEN","prob_col"):
        if k not in triage:
            raise SystemExit(f"triage_thresholds_active.json missing key: {k}")


    p = float(p)

    if triage == "RED":
        return f"High-risk pattern (RED). Consider urgent review / escalation. Risk score={p:.3f}."
    if triage == "YELLOW":
        return f"Intermediate risk (YELLOW). Recommend closer follow-up / repeat assessment. Risk score={p:.3f}."
    if triage == "GREEN":
        return f"Lower risk (GREEN). Routine follow-up suggested. Risk score={p:.3f}."

    # fallback if thresholds/labels change
    return f"Triage={triage}. Risk score={p:.3f}."
'''

def l2_normalize(X, eps=1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)

def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def warn(msg: str):
    warnings.warn(msg, RuntimeWarning, stacklevel=2)

# =========================
# Args
# =========================
ap = argparse.ArgumentParser()
ap.add_argument("--img", help="path to 2x2 montage .png")
ap.add_argument("--json_features", help="tabular features JSON")
ap.add_argument("--from_parquet", action="store_true", help="use parquet to fetch features and resolve image")
ap.add_argument("--patient", help="Patient-XXX")
ap.add_argument("--week_norm", type=int, help="normalized week integer")

# calibrated on by default, allow disabling:
ap.add_argument("--use_calibrated", dest="use_calibrated", action="store_true", default=True)
ap.add_argument("--no_calibrated", dest="use_calibrated", action="store_false", help="use raw p_stack without isotonic")

ap.add_argument("--out", help="path to write JSON output")
ap.add_argument(
    "--sot",
    default="source_of_truth/v1_tabular_only",
    help="Path to Source-of-Truth folder (default: source_of_truth/v1_tabular_only)"
)

args = ap.parse_args()

# =========================
# Paths (SINGLE SOURCE OF TRUTH)
# =========================
ROOT = Path(__file__).resolve().parents[1]
sot_arg = Path(args.sot)
SOT = (sot_arg if sot_arg.is_absolute() else (ROOT / sot_arg)).resolve()


MODELS_DIR = SOT / "models"
THRESH_DIR     = SOT / "thresholds"
DEMO_INPUTS    = SOT / "demo_inputs"

TMP_DIR = SOT / "_tmp"


# required artifacts
METADATA       = MODELS_DIR / "model_metadata.json"

mode = "multimodal"
try:
    with open(METADATA) as f:
        meta_info = json.load(f)
        mode = meta_info.get("mode", mode)
except Exception:
    pass

NEEDS_IMAGE = (mode != "tabular_only")


FEATURES_USED  = MODELS_DIR / "features_used.json"
LR_MODEL       = MODELS_DIR / "lr_base.joblib"
LGBM_MODEL     = MODELS_DIR / "lgbm_base.joblib"
STACK_MODEL    = MODELS_DIR / "stack_model.joblib"

# calibration + thresholds (active)
ISO_MODEL      = MODELS_DIR / "calibration_active.joblib"   # recommended
if not ISO_MODEL.exists():
    ISO_MODEL  = MODELS_DIR / "isotonic_stack.joblib"        # fallback

THRESH_ACTIVE  = THRESH_DIR / "triage_thresholds_active.json"

# project data (only needed for --from_parquet / montage resolving)
DATA_PATH      = ROOT / "data" / "processed" / "psp_multimodal.parquet"
PARQ           = DATA_PATH
IMGDIR         = ROOT / "report" / "montage_psp"

# =========================
# Load models
# =========================
print("[INFO] Loading models...")

if not LR_MODEL.exists():    raise FileNotFoundError(f"Missing {LR_MODEL}")
if not LGBM_MODEL.exists():  raise FileNotFoundError(f"Missing {LGBM_MODEL}")
if not STACK_MODEL.exists(): raise FileNotFoundError(f"Missing {STACK_MODEL}")

lr    = joblib.load(LR_MODEL)     # Calibrated LR pipeline
lgbm  = joblib.load(LGBM_MODEL)   # LGBM last-fit
stack = joblib.load(STACK_MODEL)  # Calibrated meta LR on [p_lr, p_lgbm]

# metadata (invert_lr)
invert_lr = False
mode = "multimodal"
try:
    with open(METADATA) as f:
        meta_info = json.load(f)
        invert_lr = bool(meta_info.get("invert_lr", False))
        mode = meta_info.get("mode", mode)
except Exception:
    pass

# thresholds (single source of truth)
if THRESH_ACTIVE.exists():
    triage = load_json(THRESH_ACTIVE)
else:
    triage = {"RED": 0.60, "YELLOW": 0.53, "GREEN": 0.40, "prob_col": "p_stack"}
    warn(f"Active thresholds not found at {THRESH_ACTIVE}. Using defaults {triage}.")
prob_col_for_triage = triage.get("prob_col", "p_stack")


# features_used (ordine e K)
features_used = None
if FEATURES_USED.exists():
    features_used = load_json(FEATURES_USED)
tab_used = (features_used or {}).get("tabular_used", None)
K = int((features_used or {}).get("pca_components", 0))

# =========================
# Resolve inputs (from_parquet shortcut)
# =========================
if args.from_parquet:
    # --- required identifiers ---
    if not args.patient or args.week_norm is None:
        raise SystemExit(
            "Usa --from_parquet insieme a --patient Patient-XXX e --week_norm N."
        )

    # --- load parquet ---
    dfp = pd.read_parquet(PARQ)
    row = dfp[
        (dfp["patient_id"].astype(str) == args.patient) &
        (dfp["week_norm"].astype(int) == int(args.week_norm))
    ]

    if row.empty:
        raise SystemExit(
            f"Nessuna riga per {args.patient} @ week_norm={args.week_norm}"
        )

    row = row.iloc[0]

    # =====================================================
    # IMAGE RESOLUTION (ONLY IF MULTIMODAL / CLIP IS USED)
    # =====================================================
    # Rule:
    # - tabular_only  -> NO image, ignore montage entirely
    # - clip_only / multimodal -> image REQUIRED

    if NEEDS_IMAGE:
        wk = row.get("week_x")
        if pd.isna(wk):
            wk = f"week-{int(row['week_norm']):03d}"

        img_path = IMGDIR / f"{args.patient}_{wk}.png"

        if not img_path.exists():
            matches = sorted(IMGDIR.glob(f"{args.patient}_week-*.png"))
            if not matches:
                raise SystemExit(
                    f"Nessuna immagine trovata per {args.patient} in {IMGDIR}"
                )
            img_path = matches[0]

        args.img = str(img_path)

    else:
        # tabular_only → no image required
        args.img = args.img or ""

    # =====================================================
    # BUILD TABULAR FEATURE JSON (ORDER GUARANTEED)
    # =====================================================
    if tab_used is None:
        raise SystemExit(
            "features_used.json mancante o invalido: impossibile "
            "ricostruire l'ordine delle feature tabellari."
        )

    feat_json = {
        c: (float(row[c]) if (c in row.index and pd.notna(row[c])) else 0.0)
        for c in tab_used
    }

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TMP_DIR / f"_tmp_export_features_{args.patient}_w{int(args.week_norm):03d}.json"

    with open(tmp, "w") as f:
        json.dump(feat_json, f, indent=2)

    args.json_features = str(tmp)


# safety check inputs
if tab_used is None:
    raise SystemExit("Impossibile determinare l'ordine delle feature tabellari (features_used.json mancante/rotto).")
if K == 0:
    # nessuna immagine necessaria
    args.img = args.img or ""
if not args.json_features:
    raise SystemExit("Manca --json_features (oppure usa --from_parquet).")

feat_json = load_json(Path(args.json_features))

# =========================
# (Optionale) load OpenCLIP/PCA solo se K>0
# =========================
clip_pca = None
if K > 0:
    import torch
    import open_clip
    # PCA model (prefer next to models, fallback in data/processed)
    PCA_CANDIDATES = [
        MODELS_DIR / "pca_clip.joblib",
        ROOT / "data" / "processed" / "pca_clip.joblib",
    ]
    pca_model_path = next((p for p in PCA_CANDIDATES if p.exists()), None)
    if pca_model_path is None:
        raise FileNotFoundError(f"PCA model not found in any of: {PCA_CANDIDATES}")
    pca = joblib.load(pca_model_path)

    if not args.img:
        raise SystemExit("Serve --img per K>0.")

    device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
    model, preprocess, _ = open_clip.create_model_and_transforms(
        'ViT-H-14', pretrained='laion2b_s32b_b79k'
    )
    model = model.to(device).eval()

    img = Image.open(args.img).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb_vec = model.encode_image(img_t).cpu().numpy().astype(np.float32)[0:1]  # (1, D)
    emb_vec = l2_normalize(emb_vec)
    clip_pca = pca.transform(emb_vec)[0]  # (K,)

# =========================
# Build feature vector (ordine esatto)
# =========================
# tabellari nell'ordine
row_vals = [(float(feat_json.get(c, 0.0)) if feat_json.get(c) is not None else 0.0) for c in tab_used]
X_tab = np.array(row_vals, dtype=np.float32).reshape(1, -1)

if K > 0:
    if len(clip_pca) != K:
        warn(f"PCA dim mismatch: expected {K}, got {len(clip_pca)}. Pad/Truncate.")
        clip_pca = (clip_pca[:K] if len(clip_pca) > K else np.pad(clip_pca, (0, K-len(clip_pca)), constant_values=0.0))
    X_clip = clip_pca.reshape(1, -1)
    X_all  = np.hstack([X_tab, X_clip])
    colnames = list(tab_used) + [f"PCA_clip_{i:02d}" for i in range(K)]
else:
    # tabular_only: nessuna colonna PCA
    X_all = X_tab
    colnames = list(tab_used)

X_df = pd.DataFrame(X_all, columns=colnames)
X_np = X_df.to_numpy(dtype=np.float32)

# =========================
# Predict (base + stack)
# =========================
p_lr = float(lr.predict_proba(X_np)[0, 1])
if invert_lr:
    p_lr = 1.0 - p_lr

p_lgbm  = float(lgbm.predict_proba(X_df)[0, 1])
p_stack = float(stack.predict_proba([[p_lr, p_lgbm]])[0, 1])

for name, v in [("p_lr", p_lr), ("p_lgbm", p_lgbm), ("p_stack", p_stack)]:
    if not (0.0 <= v <= 1.0):
        raise RuntimeError(f"{name} fuori range: {v}")


# Calibrazione isotonic (solo se serve e se richiesta)
p_stack_iso = None
if p_stack_iso is not None and not (0.0 <= p_stack_iso <= 1.0):
    raise RuntimeError(f"isotonic output fuori range: {p_stack_iso}")

use_iso = bool(args.use_calibrated) and (prob_col_for_triage in ("p_stack_iso", "p_iso"))

if use_iso:
    if ISO_MODEL.exists():
        ir = joblib.load(ISO_MODEL)
        p_stack_iso = float(ir.transform([p_stack])[0])
    else:
        warn(f"prob_col={prob_col_for_triage} ma isotonic model non trovato ({ISO_MODEL}); uso p_stack raw.")
        p_stack_iso = p_stack

# quale probabilità usare davvero per triage
if use_iso and p_stack_iso is not None:
    prob_used_for_triage = "p_stack_iso"
    p_for_triage = p_stack_iso
else:
    prob_used_for_triage = "p_stack"
    p_for_triage = p_stack


if p_for_triage >= triage["RED"]:
    tri = "RED"
elif p_for_triage >= triage["YELLOW"]:
    tri = "YELLOW"
else:
    tri = "GREEN"

def clinical_line(triage_label: str, p_used: float) -> str:
    if triage_label == "RED":
        return f"RED (High risk): consider urgent review / closer follow-up. p={p_used:.3f}"
    if triage_label == "YELLOW":
        return f"YELLOW (Intermediate): review in context (trend + imaging). p={p_used:.3f}"
    return f"GREEN (Low risk): routine follow-up. p={p_used:.3f}"


clinical = clinical_line(tri, float(p_for_triage))



# =========================
# Output
# =========================
out = {
    "p_lr": p_lr,
    "p_lgbm": p_lgbm,
    "p_stack": p_stack,
    "p_stack_iso": p_stack_iso,
    "prob_used_for_triage": prob_used_for_triage,
    "triage": tri,
    "clinical_summary": clinical,
    "meta": {
        "mode": mode,
        "n_clip_components": int(K),
        "n_tab_features": int(len(tab_used)),
        "invert_lr": bool(invert_lr),
        "thresholds_path": str(THRESH_ACTIVE if THRESH_ACTIVE.exists() else THRESH_PATH)
    }
}

js = json.dumps(out, indent=2)
print(js)
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(js)