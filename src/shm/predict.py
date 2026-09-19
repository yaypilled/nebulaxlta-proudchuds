#!/usr/bin/env python3
"""SHM inference — cumulative fatigue damage per file.

    python predict.py --input <dir-or-file> --output shm_predictions.csv
"""
import argparse, os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shm_core import cycle_histogram, predict as _predict

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory of *.csv, or a single csv")
    ap.add_argument("--output", default="shm_predictions.csv")
    ap.add_argument("--model", default=os.path.join(HERE, "model.json"))
    a = ap.parse_args()
    model = json.load(open(a.model))
    files = ([a.input] if os.path.isfile(a.input)
             else sorted(os.path.join(a.input, f) for f in os.listdir(a.input) if f.lower().endswith(".csv")))
    rows = []
    for fp in files:
        x = np.loadtxt(fp, dtype=np.float64)
        d = float(_predict(cycle_histogram(x), model)[0])
        rows.append({"file_id": os.path.basename(fp), "prediction": d})
        print(f"  {os.path.basename(fp):<16} D = {d:.6f}", file=sys.stderr)
    pd.DataFrame(rows).to_csv(a.output, index=False)
    print(f"wrote {a.output} ({len(rows)} rows)  [model: {model['kind']}]", file=sys.stderr)

if __name__ == "__main__":
    main()
