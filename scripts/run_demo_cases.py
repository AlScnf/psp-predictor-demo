import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def _extract_last_json(stdout_text: str):
    """
    Return the last valid JSON object found in stdout, even if logs are present.
    """
    decoder = json.JSONDecoder()
    best_obj = None
    best_end_pos = -1
    best_start = None
    txt = (stdout_text or "").strip()
    for i, ch in enumerate(txt):
        if ch != "{":
            continue
        try:
            obj, rel_end = decoder.raw_decode(txt[i:])
            if isinstance(obj, dict):
                end_pos = i + rel_end
                if end_pos > best_end_pos or (end_pos == best_end_pos and (best_start is None or i < best_start)):
                    best_obj = obj
                    best_end_pos = end_pos
                    best_start = i
        except json.JSONDecodeError:
            continue
    return best_obj


def run_one(export_py: Path, sot: str, patient: str, week_norm: int, use_calibrated: bool) -> dict:
    cmd = [
        sys.executable,
        str(export_py),
        "--sot", sot,
        "--from_parquet",
        "--patient", patient,
        "--week_norm", str(int(week_norm)),
    ]
    if use_calibrated:
        cmd.append("--use_calibrated")
    else:
        cmd.append("--no_calibrated")

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {
            "patient": patient,
            "week_norm": int(week_norm),
            "error": p.stderr.strip() or p.stdout.strip() or f"exit={p.returncode}",
        }

    # export_stack_v1.py may print logs before JSON.
    txt = p.stdout.strip()
    out = _extract_last_json(txt)
    if out is None:
        return {"patient": patient, "week_norm": int(week_norm), "error": "No valid JSON found in stdout", "raw": txt}

    # forza sempre id nel record
    out.setdefault("meta", {})
    out["meta"]["patient"] = patient
    out["meta"]["week_norm"] = int(week_norm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sot", default="source_of_truth/v1_tabular_only", help="Path to SoT folder")
    ap.add_argument("--cases", default="source_of_truth/v1_tabular_only/demo_inputs/demo_cases.csv", help="CSV with patient,week_norm")
    ap.add_argument("--out_json", default="report/demo_outputs/demo_results.json", help="Write aggregated JSON here")
    ap.add_argument("--out_csv", default="report/demo_outputs/demo_results.csv", help="Write quick summary CSV here")
    ap.add_argument("--no_calibrated", action="store_true", help="Use raw p_stack (no isotonic)")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    export_py = ROOT / "models" / "export_stack_v1.py"
    if not export_py.exists():
        raise SystemExit(f"Missing {export_py}")

    cases_path = (ROOT / args.cases).resolve() if not Path(args.cases).is_absolute() else Path(args.cases)
    if not cases_path.exists():
        raise SystemExit(f"Missing cases CSV: {cases_path}")

    out_json = (ROOT / args.out_json).resolve() if not Path(args.out_json).is_absolute() else Path(args.out_json)
    out_csv  = (ROOT / args.out_csv).resolve() if not Path(args.out_csv).is_absolute() else Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    use_calibrated = not args.no_calibrated

    rows = []
    with open(cases_path, newline="") as f:
        r = csv.DictReader(f)
        for line in r:
            patient = (line.get("patient") or "").strip()
            week_norm = line.get("week_norm")
            if not patient or week_norm is None:
                continue
            rows.append((patient, int(week_norm)))

    results = []
    for patient, week_norm in rows:
        res = run_one(export_py, args.sot, patient, week_norm, use_calibrated)
        results.append(res)

    # write aggregated json
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[OK] wrote {out_json}")

    # write summary csv
    # columns: patient, week_norm, triage, p_used, p_stack, p_stack_iso, clinical_summary, error
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient","week_norm","triage","prob_used_for_triage","p_stack","p_stack_iso","clinical_summary","error"])
        for res in results:
            meta = res.get("meta", {})
            patient = meta.get("patient", "")
            week_norm = meta.get("week_norm", "")
            triage = res.get("triage", "")
            prob_used = res.get("prob_used_for_triage", "")
            p_stack = res.get("p_stack", "")
            p_stack_iso = res.get("p_stack_iso", "")
            clinical = res.get("clinical_summary", "")
            err = res.get("error", "")
            w.writerow([patient, week_norm, triage, prob_used, p_stack, p_stack_iso, clinical, err])

    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
