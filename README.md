# SHM — Cumulative Fatigue Damage Regression

**Shipped model: ridge regression on physics-derived features.**
**Expected leaderboard score `0.994` (80% interval 0.992 – 0.997).**

A forecast, not a result — the organisers hold the test labels. Estimated by
grouped cross-validation and by 100 simulated 48/16 submissions in which the whole
model is re-derived from 48 files and scored on 16 it never saw.

| | Grouped CV | Simulated submission (100×) |
|---|---|---|
| **Ridge on pseudo-damage (shipped)** | **0.99507** | **0.99432** ± 0.00190 |
| 50/50 geometric blend | 0.99443 | 0.99412 ± 0.00208 |
| Physics, 2-parameter S-N law | 0.99117 | 0.99167 ± 0.00225 |

Ridge beat the physics model in 93/100 simulated submissions.

## Task framing

SHM is a **regression** task: one continuous cumulative-damage value per file. The
shipped model is a linear regression, fitted by ridge with the penalty chosen by
internal CV on the training fold.

```
log D = b₀ + Σₘ wₘ · log( Σᵢ nᵢ σᵢᵐ )        m = 2,3,…,8
```

Each feature is the rainflow **pseudo-damage** at a candidate S-N exponent. The
fitted weights are the interesting part:

| m | 2 | 3 | 4 | **5** | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| weight | +0.010 | −0.068 | +0.362 | **+0.486** | +0.316 | +0.102 | −0.082 |

They peak at m=5 and decay symmetrically — the regression independently re-derives
the physics (Miner's rule with m≈5) and then relaxes the single-exponent constraint
into a smooth blend of neighbouring power laws. Weighted-mean exponent 5.167.

The gain is **not** from capacity: 3 features (m=4,5,6) already give 0.514%, and
23 features give 0.587% — worse. There is a genuine optimum at 3–7 exponents.

## Feature extraction — where the accuracy actually came from

Two conventions dominate, both recovered empirically from the training labels.

**Repeat-history residue closure.** Rainflow counting leaves an unclosed residue —
only ~10 cycles per file, but carrying **20–74% of total σ⁵ damage**.

| Residue treatment | MAPE |
|---|---|
| discarded | 15.89% |
| counted as half-cycles (`rainflow` library default) | 2.46% |
| **closed by concatenating residue with itself** | **0.80%** |

**Sixty-four stress classes.** Binning reversals into k=64 classes — the classical
rainflow default — is an *isolated* optimum, not a trend:

| k | 48 | 56 | 60 | **64** | 68 | 72 | 80 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| MAPE | 1.33% | 1.26% | 1.19% | **0.80%** | 1.24% | 1.21% | 1.09% | 1.19% | 1.12% |

Nested CV re-selected k=64 independently in 11/11 folds.

## Model comparison

All scored on identical grouped-CV folds (`code/benchmark.py`).

| Family | Method | MAPE | Score |
|---|---|---|---|
| **Regression on physics features** | **Ridge, log-pseudo-damage m=2..8** | **0.49%** | **0.995** |
| Parametric physics | 2-parameter recovered S-N law | 0.88% | 0.991 |
| Regression on physics features | Ridge, all features pooled | 1.98% | 0.980 |
| Tree ensemble | GBoost on pseudo-damage | 9.11% | 0.909 |
| ML, no rainflow | GBoost, 24 statistical + 9 spectral | 20.00% | 0.800 |
| ML, no rainflow | Ridge, 24 statistical + 9 spectral | 23.12% | 0.769 |
| ML, no rainflow | RandomForest, same features | 30.87% | 0.691 |
| Frequency domain | Dirlik spectral method | 31.83% | 0.682 |
| Frequency domain | Narrow-band approximation | 45.10% | 0.549 |

Two readings worth stating in a write-up. Generic ML on raw signal statistics
plateaus around 20% MAPE — the target is a σ⁵-weighted functional that statistical
summaries cannot represent. But regression *on top of* physics-derived features beats
the pure physics model. The feature engineering carries the information; the
regression then exploits the residual flexibility the single-exponent law forbids.

## Tested and rejected

| Variant | Result |
|---|---|
| Endurance cutoff σ_cut ∈ {0.5 … 15} | 0.797% vs 0.802% — noise |
| Dual-slope / Haibach knee (8 knees × 6 deltas) | 0.801% vs 0.802% — noise |
| Goodman mean-stress, Su ∈ {200 … 10000} | monotonically → no-correction limit |
| Walker mean-stress, γ ∈ {0.5 … 1.0} | monotonically → γ=1 (no correction) |
| Linear mean-stress, a ∈ {−0.01 … +0.01} | monotonically → a=0 (no correction) |
| Nonparametric S-N, 2–8 knot log-log spline | LOO selected the 2-knot (power-law) solution |
| Residual correction from 24 signal features | CV R² = 0.075 — not predictable, excluded |

Each mean-stress family degrades *monotonically* away from its own no-correction
limit — strong evidence the reference used pure stress range.

## Train/validation split

No official split is given and file numbering is explicitly random. The data spans
two lines × two load conditions (AW0/AW4), so a random split risks leaking
condition-level structure.

Groups were built **without labels**: 24 statistical + spectral features →
standardise → PCA(4) → KMeans(4), any group under 4 files merged into its nearest
neighbour. Result: 3 groups of 22 / 28 / 14. GroupKFold over these is the headline,
with the model refit from scratch inside every fold.

Grouped CV (0.493%) and random CV (0.580%) agree closely — no condition-level
leakage available to exploit.

## Leakage and overfitting checks

| Check | Result |
|---|---|
| Fold partitions differ between seeds | verified, all distinct |
| Permutation test (labels shuffled) | MAPE 118% vs 0.58% real — **204× collapse** |
| Stability over 20 reshuffles | sd 0.016% |
| Nested CV, config re-searched per fold | 0.905% vs 0.883% — 0.02pp selection bias |
| Simulated 48/16 submission × 100 | 0.99432 ± 0.00190 |
| Capacity sweep | more features make it worse — not overfitting |
| Test features inside training envelope | **16/16 on all 7 features** — interpolating |

Not measured by any of the above: the 64 labels were visible while deciding *what to
try*. The automated search is priced in; human choices cannot be. Mitigation is that
the key discovery (residue carries most of the damage) is a label-free property of
the signal, and the fix is textbook practice rather than a tuned knob.

## Residual risk

Two test files (test03 at 73.8%, test02 at 62.3%) exceed the training maximum
residue share of 62.0%, so their answers lean hardest on the closure convention
matching the organisers'. Mitigating: residue share does not predict error on
training data (r = 0.07), and the ridge and physics models agree to within 0.45% and
0.24% on exactly those two files.

## Files

```
shm_predictions.csv         16 rows, file_id + prediction
predictions.zip             flat, ready to submit
code/shm_core.py            feature extraction + both models
code/predict.py             CLI: --input <dir|file> --output shm_predictions.csv
code/benchmark.py           reproduces the model-comparison table
code/evaluate.py            reproduces the validation numbers
code/judge_leaderboard.py   replica of the organisers' scorer
code/model.joblib           fitted pipeline
code/model.json             readable coefficients
```

```bash
python code/predict.py --input <test_dir> --output shm_predictions.csv
python code/judge_leaderboard.py shm_predictions.csv <truth.csv>
```

## Caveats

- The scorer is a replica built from the disclosed formula; the organisers'
  `judge_leaderboard.py` is not in the repo.
- Simulated-submission figures assume the 16 test files resemble the 64 training
  files. Feature-envelope checks support this but do not prove it.
- Per-file error on held-out folds: median 0.27%, 90th percentile 0.86%, worst 5.66%.
