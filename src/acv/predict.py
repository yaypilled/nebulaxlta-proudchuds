#!/usr/bin/env python3
"""ACV inference — rank each car from most to least likely to have a refrigerant leak.

The frozen ensemble of six equally-weighted components (notebook section 17):
  domain percentile score, logistic regression, random forest, extra trees,
  and 30- and 60-minute persistence scores.

Library use (what the web app calls):

    from acv.predict import load_bundle, rank_cars, ranked_cars_string
    bundle = load_bundle("artifacts/acv/model_bundle.joblib")
    ranking = rank_cars("acv_test_case.xlsx", bundle)     # DataFrame, best first
    s = ranked_cars_string(ranking)                        # "05|02|07|..."

Command line (mirrors the other subsystems):

    python -m acv.predict --input <file-or-dir> --output acv_predictions.csv
"""
import argparse, os, sys, re
import joblib, numpy as np, pandas as pd

from .features import (extract_case_features, add_domain_score, normalized_rank_score,
                       window_aggregate, LOCKED_METHOD_WEIGHTS)

DEFAULT_BUNDLE = os.path.join("artifacts", "acv", "model_bundle.joblib")


def load_bundle(path=DEFAULT_BUNDLE):
    """Load the frozen bundle saved by acv.train (or by the notebook)."""
    b = joblib.load(path)
    b.setdefault("ensemble_weights", LOCKED_METHOD_WEIGHTS)
    b.setdefault("model_features", b.get("robust_features"))
    return b


def rank_cars(source, bundle):
    """source: path to an .xlsx, or an already-loaded DataFrame.
    Returns a DataFrame sorted best-first with per-method and ensemble scores."""
    raw = source if isinstance(source, pd.DataFrame) else pd.read_excel(source)
    feats = extract_case_features(raw)
    if feats.empty:
        raise ValueError("No 'Car NN - ...' columns found; is this an ACV telemetry file?")

    weights = bundle["ensemble_weights"]
    model_features = bundle["model_features"]
    medians = bundle["training_medians"]

    result = feats[["car"]].copy()

    # 1. domain percentile consensus
    result["domain"] = normalized_rank_score(add_domain_score(feats)["domain_score"]).values

    # 2. supervised models, imputed with TRAINING medians (never the test file's own)
    X = feats[model_features].fillna(medians).fillna(0)
    for name, model in bundle["models"].items():
        result[name] = normalized_rank_score(model.predict_proba(X)[:, 1]).values

    # 3. persistence across 30- and 60-minute windows
    for minutes, name in [(30, "window30"), (60, "window60")]:
        ws = window_aggregate(raw, minutes=minutes)
        merged = result[["car"]].merge(ws[["car", "window_score"]], on="car", how="left")
        result[name] = normalized_rank_score(merged["window_score"].fillna(0.5)).values

    result["ensemble_score"] = sum(weights[n] * result[n] for n in weights)
    result = result.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
    result["predicted_rank"] = np.arange(1, len(result) + 1)

    top = result.iloc[0]["car"]
    result.attrs["top_car"] = top
    result.attrs["margin"] = float(result.iloc[0]["ensemble_score"] - result.iloc[1]["ensemble_score"]) \
        if len(result) > 1 else float("nan")
    result.attrs["agreement"] = sum(
        result.loc[result[m].idxmax(), "car"] == top for m in weights)
    result.attrs["n_methods"] = len(weights)
    return result


def ranked_cars_string(ranking):
    """The submission field: car ids best-first, pipe separated, e.g. '05|02|07'."""
    cars = ranking["car"].tolist()
    if len(set(cars)) != len(cars):
        raise ValueError("duplicate car in ranking")
    if not all(re.fullmatch(r"\d{2}", c) for c in cars):
        raise ValueError(f"car ids must be two digits as they appear in the headers: {cars}")
    return "|".join(cars)


def predict_files(paths, bundle):
    rows = []
    for p in paths:
        ranking = rank_cars(p, bundle)
        fid = os.path.basename(p).replace("(1)", "")
        rows.append({"file_id": fid, "ranked_cars": ranked_cars_string(ranking)})
        print(f"  {fid:<24} {rows[-1]['ranked_cars']}"
              f"   (top={ranking.attrs['top_car']}, "
              f"agreement={ranking.attrs['agreement']}/{ranking.attrs['n_methods']}, "
              f"margin={ranking.attrs['margin']:.4f})", file=sys.stderr)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="an .xlsx, or a directory of them")
    ap.add_argument("--output", default="acv_predictions.csv")
    ap.add_argument("--model", default=DEFAULT_BUNDLE)
    a = ap.parse_args()

    bundle = load_bundle(a.model)
    if os.path.isfile(a.input):
        paths = [a.input]
    else:
        paths = sorted(os.path.join(a.input, f) for f in os.listdir(a.input)
                       if f.lower().endswith((".xlsx", ".xls")) and not f.startswith("~$"))
    if not paths:
        raise SystemExit(f"no spreadsheets found in {a.input}")

    predict_files(paths, bundle).to_csv(a.output, index=False)
    print(f"wrote {a.output} ({len(paths)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
