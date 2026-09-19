# Train Condition Intelligence

One Streamlit application for all four LTA × NebulaX PS3 tasks: Door cycle detection, ACV fault localisation, Rail Corrugation classification and cumulative fatigue-damage estimation.

**Deployment status: not deployed. No live Cloud Run URL is available yet.** The target project supplied by the team is `qwiklabs-gcp-02-d9853c9ea2f4`. This build environment has no authenticated Google Cloud CLI or ADC, and its cloud browser cannot open the supplied project console. No cloud resource has been created. Docker is not installed here, so image build and Cloud Run startup remain unverified.

## Run locally

Python 3.12 is the tested runtime. In a checkout of the `unified-app` branch:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
streamlit run app.py --server.address=127.0.0.1 --browser.serverAddress=localhost
```

The ordinary launch command is `streamlit run app.py`. The explicit localhost settings above prevent Streamlit's optional public-IP lookup during local startup.

Production inference requires only application code, the committed model artifacts and uploaded input files. Training data are not loaded at startup. Models are loaded once and cached. Predictions remain within the current Streamlit session; temporary input files are removed after processing.

## Subsystems

| Subsystem | Production source | Input | Official output | Status |
|---|---|---|---|---|
| Door | QH, `qh-door` | One continuous CSV | `start_time,end_time,prediction` | Operational locally |
| ACV | `ACV`, fixed fitted bundle | Standard-schema XLSX | `file_id,ranked_cars` | Operational locally |
| Rail Corrugation | `overnight-writeup` / `qh` | CSV, 10,000 rows × 129 columns | `file_id,prediction` | Operational locally |
| SHM | `bryan`, reconstructed fitted ridge | Headerless single-column stress CSV | `file_id,prediction` | Operational locally |

Select a subsystem, upload telemetry, click **Analyse**, inspect the result and download the official CSV. The **Submission Centre** collects the generated outputs into `predictions.zip`, containing only the official CSVs at the archive root. Schema validation runs before download. A changed or failed upload invalidates the affected result. Other subsystem results remain available.

Do not submit outputs generated from training files. Source filenames are shown in the Submission Centre so the team can confirm it used the held-out inputs.

## Model integration

- **Door:** preserves QH's 1.0-second gap segmentation and committed Open/Close current-sum thresholds. Ambiguous operation flags use the model's fitted global threshold and generate a warning. Timestamps are echoed from the input. The model threshold is not a universal safety limit. Andre's door-position regressor is not used.
- **ACV:** production code is extracted from the notebook, with no inference-time fitting. All eight car IDs come from the uploaded headers. The six fixed ranking components use the saved bundle and its training medians. The richer case-04 schema is explicitly rejected because it is outside the saved model's validated scope. Scores are suspicion rankings, not calibrated failure probabilities.
- **Rail:** retains the fitted model and feature extraction. Importing inference no longer resolves the developer's training directory. The documented low-speed rule is shown separately from processing failures. File shape and finite values are validated before applying that rule. Real processing failures prevent a downloadable result.
- **SHM:** the source JSON omitted the fitted scaler and intercept. `scripts/reconstruct_shm.py` reconstructs the complete ridge pipeline from the 64 official training files, then compares predictions with the committed output as a reproducibility check. It does not use hidden test labels. All 16 committed predictions agree within a maximum relative difference of `1.23e-10`. The task is cumulative damage, not remaining useful life.

Model artifacts are under `artifacts/`. The original Rail and ACV binaries are preserved. The Rail artifact uses scikit-learn 1.9.1, while ACV was saved with 1.8.0. The pinned unified runtime uses 1.9.1, which emits the known ACV version warning. Compatibility was checked on the five standard training cases and the test case against a separate 1.8.0 runtime: all six rankings and every component score agree to `1e-12`. See `docs/integration/acv_runtime_parity.json`; no model was retrained for that comparison.

## Verification

```bash
python -m pytest -q
```

54 contract and metric tests pass, including the preserved Door metric tests and the concurrent-analysis guard. The actual Streamlit uploader and Analyse button were exercised with every official test input:

| Subsystem | Measured result | Local inference time |
|---|---|---:|
| Door | 6,253 input rows; 38 cycles; 28 Normal / 10 Abnormal resistance; 18 Open / 20 Close; no ambiguous operation | 0.14 s |
| ACV | `01\|03\|04\|07\|08\|06\|02\|05` | 24.33 s |
| Rail | 68 files; 57 Normal / 5 Side I / 6 Side II; 10 low-speed rule results; no failure fallback accepted | 80.57 s |
| SHM | 16 files; all finite numeric outputs | 7.85 s |

These are predictions and runtime measurements, not held-out accuracy claims. The organisers retain the test answers. Local HTTP checks returned 200 for both `/` and `/_stcore/health`. Streamlit AppTest verified all pages, real uploads, result rendering, download controls, archive generation and a friendly malformed-upload error that clears stale output. Screenshot/browser visual review is still pending because the cloud browser cannot reach this environment's localhost server.

Reproduce the complete app test with raw data stored outside the repository:

```bash
python -m scripts.verify_app \
  --data /absolute/path/to/02_Datasets \
  --acv-test /absolute/path/to/acv_test_case.xlsx \
  --output outputs/official \
  --full
python scripts/validate_submission.py outputs/official/predictions.zip
```

Door training-boundary and rolling-origin evaluation:

```bash
NEBULAX_DATA=/absolute/path/to/02_Datasets python -m src.door.evaluate
```

This reproduces 110 exact training boundaries, five rolling-origin IoU-weighted F1 scores of 1.000, and an always-Normal baseline of 0.7273. The small sample limits generalisation. Additional leave-one-out/reverse-order claims are intentionally not advertised.

SHM artifact reconstruction:

```bash
python -m scripts.reconstruct_shm --data /absolute/path/to/02_Datasets/SHM
```

Historical Rail methodology and evaluation logs are preserved under `docs/`; they describe prior subsystem experiments. Current integration evidence is in `docs/integration/`.

## Google Cloud Run deployment

Target:

- Project: `qwiklabs-gcp-02-d9853c9ea2f4`
- Region: `asia-southeast1`
- Service: `lta-nebulax-ps3`
- Live URL / deployed revision: **not available, deployment blocked by access**

From an authenticated Google Cloud Shell or terminal with this repository and Python dependencies installed:

```bash
python -m scripts.deploy_cloud_run --project qwiklabs-gcp-02-d9853c9ea2f4
```

The script checks the active login/project and local tests before enabling Cloud Run, Cloud Build and Artifact Registry. It deploys the Dockerfile, obtains the actual URL, checks the HTTP page and health endpoint, and reads recent runtime logs. It writes the real service URL and revision to `outputs/deployment.json`. Browser loading and a live upload/inference/download must still be verified before declaring completion.

Exact equivalent redeploy command, after authentication and API setup:

```bash
gcloud run deploy lta-nebulax-ps3 \
  --project qwiklabs-gcp-02-d9853c9ea2f4 \
  --source . \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 2 \
  --concurrency 8 \
  --cpu 2 \
  --memory 8Gi \
  --timeout 3600 \
  --port 8080 \
  --session-affinity
```

The full Streamlit test peaked at **4,003.7 MiB** of process memory while uploading all 68 Rail files together (about 1.18 GB of raw input). A 4-GiB instance leaves insufficient headroom for the complete upload workflow and temporary-file memory. The initial setting is therefore 8 GiB / 2 CPUs and a maximum of two instances. Minimum instances are zero. HTTP concurrency is eight so the Streamlit WebSocket can stay open while uploads and downloads run. The application permits one active analysis per process and asks a concurrent caller to retry. Large simultaneous uploads still require capacity testing beyond this hackathon demonstration. No database, GPU, VM or cluster is required. Re-measure on Cloud Run before reducing memory. Streamlit sessions are in memory; session affinity is best effort, and a restart can discard the session's outputs, so download completed results promptly.

The container listens on `0.0.0.0` and Cloud Run's `PORT`. It contains no raw training dataset. IAM permissions, API availability, lab quotas and public invocation policy still require verification in the actual project. Do not create a new project or add credentials to Git to work around an access failure.

## Suggested judging video (2 minutes 50 seconds)

1. **0:00–0:15:** Overview, four operational subsystems and the maintenance workflow.
2. **0:15–0:45:** Upload Door Test.csv, Analyse, point out the 38 cycles and abnormal-cycle timeline, download its CSV.
3. **0:45–1:20:** Upload ACV test XLSX, Analyse, show Car 01 and the complete ranking. Explain that the scores prioritise inspection.
4. **1:20–1:50:** Upload two Rail examples, Analyse, show the side labels and any low-speed rule notice. Full-test outputs can already have been generated for submission.
5. **1:50–2:20:** Upload SHM stress files, Analyse, show cumulative damage and the optional stress trace.
6. **2:20–2:50:** Open Submission Centre, show validated CSVs and download predictions.zip. State the small-data/model limitations briefly.

Keep a separate final submission archive generated from the complete held-out test set; a short demonstration batch is not a substitute for all 68 Rail and 16 SHM test predictions.
