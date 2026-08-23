# MIRROR-Rad

**Mutual Inference and Reciprocal Retrieval for Omission and Risk Detection in Radiology Summaries**

MIRROR-Rad is an evidence-grounded clinical NLP application that audits whether a candidate radiology summary is faithful to a supplied source report. It checks the text in both directions so that added or contradicted claims and omitted source findings are handled separately, then exposes the evidence, clinical mismatch signals, confidence, and review priority behind each decision.

> Research use only. The application compares supplied text with supplied text. It does not inspect radiographs, establish clinical truth, diagnose disease, recommend treatment, or replace qualified review. A public deployment must receive only synthetic or de-identified text—never PHI or identifiable patient data.

## What is included

- Fine-tuned BioClinicalBERT three-way NLI for `ENTAILED`, `CONTRADICTED`, and `UNSUPPORTED` claims.
- Reciprocal candidate-to-source and source-to-candidate evidence retrieval with both raw and clinical-slot-masked TF-IDF matching.
- Entity-bound semantic-delta checks for negation, laterality, anatomy, severity, temporal status, measurements, device identity, device position, uncertainty, and entity substitution.
- BioClinicalBERT continued adaptation on patient-disjoint ReXErr training roles with Open-I replay; gradient training, fusion fitting, calibration, and adaptation evaluation remain separate.
- Multi-domain learned claim fusion, held-out calibration, abstention, count-aware error-mass aggregation, and complete-report review routing.
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

The bundled aggregate ledger retains the measured evaluation populations, metrics, confidence intervals, baselines, ablations, and validity boundaries while intentionally excluding dataset rows and report text. The expert-family-quarantined controlled evaluation reports macro-F1 **0.9623**, contradiction recall **0.9538**, unsupported-claim recall **1.0000**, high-risk unsafe recall **0.9928**, and expected calibration error **0.0189**. These values come from controlled counterfactual claims and must not be interpreted as overall accuracy or hospital performance.

External evaluation is deliberately separated:

- The final patient-disjoint adaptation-set rerun reports macro-F1 **0.9010**, contradiction recall **0.9150**, unsupported recall **0.8098**, and ECE **0.0077**. The original single-open result was retained, but a later corrective multi-domain fusion fit means the displayed final-artifact rerun is iterative evidence rather than a new pristine holdout.
- ReXErr measures transfer to clinician-designed synthetic/silver semantic errors. The contextual sentence analysis reports AUROC **0.9943**, AUPRC **0.9914**, recall **0.9811**, and specificity **0.9698**. Its stricter exact-pair-decontaminated slice reports AUROC **0.9911**, AUPRC **0.9885**, recall **0.9817**, and specificity **0.9506**.
- On the strict unseen-pair ReXErr slice, recall is **0.9506** for device-name changes, **0.9976** for device-position changes, and **0.9754** for severity changes—the categories that motivated the upgrade.
- Complete-report ReXErr routing reports AUROC **0.9438**, AUPRC **0.9991**, recall **0.9574**, and specificity **0.7059**. Its separate matched-identity false-alert rate is **0.1211**.
- RadEvalX measures report-level ranking against a small two-radiologist consensus benchmark. The declared count-aware score reports AUROC **0.6740**, AUPRC **0.7572**, and captures **29.8%** of significant error burden at a 20% review budget. CheXbert reports AUROC **0.7415** on the same cohort; the paired MIRROR-Rad-minus-CheXbert AUROC interval is **−0.2154 to 0.0723**, so the apparent difference is not statistically resolved on 100 reports.
- Natural claim-level accuracy, clinical effectiveness, and multi-site generalization are not claimed.

The external ReXErr and RadEvalX results are explicitly labelled iterative because earlier development inspected those benchmarks. MIRROR-Rad does not hide the weaker RadEvalX result or describe the controlled score as “97% overall accuracy.”

## Deployment contents

Only operational files are published: the Streamlit entrypoint and theme, Python engine, exact dependency declarations, trained checkpoint and tokenizer, calibrated fusion and routing artifacts, integrity manifest, aggregate evaluation ledger, license, and third-party notices. Datasets, report text, benchmark prediction rows, annotation packets, generated summaries, workflow posters, blueprints, research reports, training scripts, tests, caches, local environments, and development outputs are excluded by a positive allowlist.

## License and upstream model

Project code is provided under the MIT License. The included checkpoint is fine-tuned from [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT), whose upstream model repository declares the MIT License. See `THIRD_PARTY_NOTICES.md` for attribution and distribution boundaries. No source dataset is distributed in this repository.
