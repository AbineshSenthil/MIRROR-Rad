# Third-party notices

MIRROR-Rad is distributed under the MIT License. Its runtime uses third-party open-source software; each component remains governed by its own license. The pinned dependency graph is recorded in `requirements-lock.txt`.

Key components include PyTorch, Transformers, BioClinicalBERT-derived model assets, scikit-learn, Streamlit, pandas, NumPy, SciPy, Plotly, and their transitive dependencies. The upstream `emilyalsentzer/Bio_ClinicalBERT` model repository and model card identify the model license as MIT. The included fine-tuned checkpoint retains that attribution and should cite Alsentzer et al., “Publicly Available Clinical BERT Embeddings,” ClinicalNLP 2019. Upstream references: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT and https://github.com/EmilyAlsentzer/clinicalBERT.

Dataset text, radiographs, row-level predictions, study identifiers, and report-level audit traces are not part of the public deployment repository. Only aggregate evaluation evidence is bundled. Open-I/IU X-Ray, ReXErr, and RadEvalX remain subject to their source terms and citation requirements; obtain them from their official distributors for independent research.
