# MIRROR-Rad

**Mutual Inference and Reciprocal Retrieval for Omission and Risk Detection in Radiology Summaries**

MIRROR-Rad is an evidence-grounded clinical NLP application that audits whether a candidate radiology summary is faithful to a supplied source report. It checks the text in both directions so that added or contradicted claims and omitted source findings are handled separately, then exposes the evidence, clinical mismatch signals, confidence, and review priority behind each decision.

> Research use only. The application compares supplied text with supplied text. It does not inspect radiographs, establish clinical truth, diagnose disease, recommend treatment, or replace qualified review. A public deployment must receive only synthetic or de-identified text—never PHI or identifiable patient data.

## What is included

- Fine-tuned BioClinicalBERT three-way NLI for `ENTAILED`, `CONTRADICTED`, and `UNSUPPORTED` claims.
- Reciprocal candidate-to-source and source-to-candidate evidence retrieval.
- Explicit checks for negation, laterality, anatomy, severity, temporal status, measurements, uncertainty, and entity substitution.
- Learned claim-level signal fusion, post-hoc calibration, abstention, and report-level review routing.
- Single-report auditing, guided live examples, CSV/JSON/JSONL dataset upload, bounded batch analysis, and CSV/JSON assurance exports.
- Aggregate evaluation evidence without raw reports, generated summaries, claims, study identifiers, or row-level prediction ledgers.
- Fail-closed startup validation for application code, model weights, tokenizer assets, calibration, fusion, routing, and evaluation evidence.

## Deploy manually on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
2. Select repository `AbineshSenthil/MIRROR-Rad`, branch `main`, and entrypoint `app.py`.
3. Open **Advanced settings** and select **Python 3.11**.
4. No API key, Hugging Face token, or other secret is required. The trained checkpoint is bundled through Git LFS and inference is offline.
5. Deploy and allow extra time for the first build to retrieve the approximately 413 MiB LFS checkpoint and load the CPU model.

The repository already contains the root `requirements.txt`, Debian `packages.txt`, `.streamlit/config.toml`, and exact model assets required by Community Cloud. Streamlit officially supports Git LFS-backed repository files.

## Dataset upload contract

The Dataset Lab accepts UTF-8 `.csv`, `.json`, and `.jsonl` files.

| Field | Requirement | Meaning |
|---|---|---|
| `findings` | Required | Source report or Findings text used as evidence |
| `impression` | Required | Candidate summary or Impression to audit |
| `case_id` | Optional | Stable case identifier; generated when absent |
| `generator` | Optional | Summary-generator label for grouped review |
| `site` | Optional | De-identified source/site label |
| `split` | Optional | Dataset partition label |
| `subgroup` | Optional | De-identified monitoring group |

Public deployment defaults are 10 MB and 100 validated report pairs per upload, with CPU-safe report and NLI batches of eight. Uploaded content remains in the running application session; on an online deployment it is still transmitted to the hosting service before the application can screen it. The acknowledgement and privacy preflight reduce accidental misuse but are not HIPAA controls.

## Local run

Install Git LFS before cloning so the checkpoint is materialized instead of remaining a pointer file.

```powershell
git lfs install
git clone https://github.com/AbineshSenthil/MIRROR-Rad.git
cd MIRROR-Rad
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --requirement requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://127.0.0.1:8501/`. CPU inference is supported. The application serializes inference and caches one model instance to stay within constrained hosting environments.

## Evidence and interpretation

The bundled aggregate ledger retains the measured evaluation populations, metrics, confidence intervals, baselines, ablations, and validity boundaries while intentionally excluding dataset rows and report text. The headline controlled evaluation reports macro-F1 0.9505, contradiction recall 0.9785, unsupported-claim recall 0.9412, high-risk unsafe recall 0.9928, and expected calibration error 0.0281. These values come from controlled counterfactual claims and must not be interpreted as hospital performance.

External evaluation is deliberately separated:

- ReXErr measures transfer to clinician-designed synthetic/silver semantic errors.
- RadEvalX measures report-level ranking against a small two-radiologist consensus benchmark.
- Natural claim-level accuracy, clinical effectiveness, and multi-site generalization are not claimed.

## Deployment contents

Only operational files are published: the Streamlit entrypoint and theme, Python engine, exact dependency declarations, trained checkpoint and tokenizer, calibrated fusion and routing artifacts, integrity manifest, aggregate evaluation ledger, license, and third-party notices. Datasets, report text, benchmark prediction rows, annotation packets, generated summaries, workflow posters, blueprints, research reports, training scripts, tests, caches, local environments, and development outputs are excluded by a positive allowlist.

## License and upstream model

Project code is provided under the MIT License. The included checkpoint is fine-tuned from [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT), whose upstream model repository declares the MIT License. See `THIRD_PARTY_NOTICES.md` for attribution and distribution boundaries. No source dataset is distributed in this repository.

