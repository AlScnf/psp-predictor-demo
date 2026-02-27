import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb

RISKY_SUBSTRINGS = ["psp","tier","label_","ground","gt"]
RISKY_EXACT = set(["psp_score","psp_label","target_psp","tier","week","week_lab","week_ord"])
ALLOW_PREFIX = ("PCA_clip_",)
ALLOW_EXACT = set([
    "enh_ml","edema_ml","whole_ml",
    "enh_ml_dpct_prev","edema_ml_dpct_prev","whole_ml_dpct_prev",
    "enh_ml_dpct_base","edema_ml_dpct_base","whole_ml_dpct_base",
    "week_norm_bin"
])

def is_risky(c):
    lc = c.lower()
    if c in RISKY_EXACT: return True
    if any(s in lc for s in RISKY_SUBSTRINGS):
        return c != "label"
    return False

def is_allowed(c): return c.startswith(ALLOW_PREFIX) or (c in ALLOW_EXACT)

def select_feats(df, idcols, ycol):
    drop = set(idcols + [ycol, "montage_path"])
    feats = [c for c in df.columns
             if c not in drop and pd.api.types.is_numeric_dtype(df[c]) and not is_risky(c) and is_allowed(c)]
    return feats

def ece_score(y, p, n_bins=10):
    bins = np.linspace(0,1,n_bins+1); idx=np.digitize(p,bins)-1
    ece=0.0
    for b in range(n_bins):
        m=(idx==b)
        if m.sum()==0: continue
        ece += (m.mean())*abs(y[m].mean()-p[m].mean())
    return float(ece)

def metrics_from(y, p):
    return {"AUROC": float(roc_auc_score(y,p)),
            "AUPRC": float(average_precision_score(y,p)),
            "Brier": float(brier_score_loss(y,p)),
            "ECE(10)": float(ece_score(y,p,10))}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--idcols", nargs="+", default=["patient_id","week_norm"])
    ap.add_argument("--label", default="label")
    ap.add_argument("--outdir", default="report/psp")
    ap.add_argument("--nan_col_thresh", type=float, default=0.4)
    ap.add_argument("--lgbm_num_leaves", type=int, default=3)
    ap.add_argument("--lgbm_max_depth", type=int, default=-1)
    ap.add_argument("--lgbm_lr", type=float, default=0.01)
    ap.add_argument("--lgbm_estimators", type=int, default=1200)
    ap.add_argument("--lgbm_colsample", type=float, default=0.9)
    ap.add_argument("--lgbm_subsample", type=float, default=0.9)
    return ap.parse_args()

def main():
    args = parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.data).copy()
    ycol = args.label; df[ycol] = pd.to_numeric(df[ycol]).astype(int)

    feats = select_feats(df, args.idcols, ycol)
    if not feats: raise SystemExit("No features after gating.")
    nan_frac = df[feats].isna().mean()
    feats = [c for c in feats if nan_frac[c] <= args.nan_col_thresh]
    if not feats: 
        raise SystemExit("All features exceed NaN threshold.")

    patients = np.sort(df["patient_id"].astype(str).unique())

    # LR pipe
    lr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    # LGBM small-N tuned
    lgbm_params = dict(
        objective="binary",
        learning_rate=args.lgbm_lr,
        num_leaves=args.lgbm_num_leaves,
        max_depth=args.lgbm_max_depth,
        n_estimators=args.lgbm_estimators,
        subsample=args.lgbm_subsample,
        colsample_bytree=args.lgbm_colsample,
        reg_lambda=1.0,
        reg_alpha=0.0,
        n_jobs=-1,
        verbose=-1
    )
    lgbm = lgb.LGBMClassifier(**lgbm_params)

    oof_lr  = np.zeros(len(df), dtype=float)
    oof_lgb = np.zeros(len(df), dtype=float)
    y_true  = df[ycol].to_numpy(int)

    for pid in patients:
        te = (df["patient_id"].astype(str)==pid); tr = ~te
        Xtr_df, ytr = df.loc[tr, feats].copy(), df.loc[tr, ycol].to_numpy().astype(int)
        Xte_df, yte = df.loc[te, feats].copy(), df.loc[te, ycol].to_numpy().astype(int)

        lr_cal = CalibratedClassifierCV(lr_pipe, method="sigmoid", cv=3)
        lr_cal.fit(Xtr_df.to_numpy(), ytr)
        oof_lr[te] = lr_cal.predict_proba(Xte_df.to_numpy())[:,1]

        pos = ytr.sum(); neg = len(ytr)-pos
        spw = (neg / max(pos,1))
        lgbm.set_params(scale_pos_weight=spw)
        lgbm.fit(Xtr_df, ytr, eval_set=[(Xtr_df, ytr)], eval_metric="auc")
        oof_lgb[te] = lgbm.predict_proba(Xte_df)[:,1]

    base_df = df[args.idcols + [ycol]].copy()
    base_df["p_lr"] = oof_lr
    base_df["p_lgbm"] = oof_lgb
    base_df.to_csv(out/"preds_oof_base.csv", index=False)

    m_lr, m_lgbm = metrics_from(y_true, oof_lr), metrics_from(y_true, oof_lgb)

    Xmeta = np.vstack([oof_lr, oof_lgb]).T
    meta_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced")
    )
    meta_cal = CalibratedClassifierCV(meta_pipe, method="sigmoid", cv=3)
    meta_cal.fit(Xmeta, y_true)
    p_stack = meta_cal.predict_proba(Xmeta)[:,1]
    m_stack = metrics_from(y_true, p_stack)

    rng = np.random.default_rng(42)
    y_rand = rng.permutation(y_true)
    m_rand = {"AUROC_randomLabel": float(roc_auc_score(y_rand, p_stack)),
              "AUPRC_randomLabel": float(average_precision_score(y_rand, p_stack))}

    meta_df = base_df.copy(); meta_df["p_stack"]=p_stack
    meta_df.to_csv(out/"preds_stack_in_sample.csv", index=False)
    metrics = {"LR": m_lr, "LGBM": m_lgbm, "STACK": m_stack}; metrics.update(m_rand)
    with open(out/"metrics_stack_loocv.json","w") as f: json.dump(metrics, f, indent=2)


    # ------------------------------
    # FINAL FIT (train on full data) + SAVE MODELS
    # ------------------------------
    import joblib

    Xfull_df = df[feats].copy()
    yfull = y_true

    # Fit calibrated LR on full data
    lr_cal_final = CalibratedClassifierCV(lr_pipe, method="sigmoid", cv=3)
    lr_cal_final.fit(Xfull_df.to_numpy(), yfull)

    # Fit LGBM on full data
    pos = int(yfull.sum())
    neg = int(len(yfull) - pos)
    spw = (neg / max(pos, 1))

    lgbm_final = lgb.LGBMClassifier(**lgbm_params)
    lgbm_final.set_params(scale_pos_weight=spw)
    lgbm_final.fit(Xfull_df, yfull, eval_set=[(Xfull_df, yfull)], eval_metric="auc")

    # Meta features on full data
    p_lr_full = lr_cal_final.predict_proba(Xfull_df.to_numpy())[:, 1]
    p_lgb_full = lgbm_final.predict_proba(Xfull_df)[:, 1]
    Xmeta_full = np.vstack([p_lr_full, p_lgb_full]).T

    # Fit calibrated meta model on full data
    meta_cal_final = CalibratedClassifierCV(meta_pipe, method="sigmoid", cv=3)
    meta_cal_final.fit(Xmeta_full, yfull)

    models_dir = Path(args.outdir) / "models"
    models_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(lr_cal_final, models_dir / "lr_base.joblib")
    joblib.dump(lgbm_final, models_dir / "lgbm_base.joblib")
    joblib.dump(meta_cal_final, models_dir / "stack_model.joblib")

    # Save the feature list used by the final model
    with open(models_dir / "features_used.json", "w") as f:
        json.dump({"features": feats}, f, indent=2)

    print(f"[OK] Saved FINAL models → {models_dir}")



    print("Base LR:", m_lr)
    print("Base LGBM:", m_lgbm)
    print("STACK:", m_stack)
    print("Random-label sanity:", m_rand)

if __name__ == "__main__":
    main()
