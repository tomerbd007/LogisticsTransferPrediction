# 🚚 Regional Logistics Transfer — Delay Prediction

> **Will a logistics transfer arrive more than 30 minutes late?**
> An end‑to‑end machine‑learning project (Intro to Machine Learning, 2026 — **Group #15**) that takes
> raw operational transfer records from a regional distribution network and predicts the probability
> of a significant arrival delay.

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white">
<img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white">
<img alt="Metric" src="https://img.shields.io/badge/metric-ROC--AUC-2E8B57">
<img alt="Task" src="https://img.shields.io/badge/task-binary%20classification-555">
</p>

---

## 📋 Problem

| | |
|---|---|
| **Goal** | Predict `is_delayed` — a transfer arriving **> 30 min late** vs. the scheduled time |
| **Type** | Binary classification (probability output) |
| **Metric** | **ROC‑AUC** (threshold‑free, robust to class imbalance) |
| **Data** | **51,056** transfers · **22** raw predictors → **52** engineered features |
| **Imbalance** | **~14%** delayed (positive class) — handled with class weighting |

The whole solution lives in a single, fully‑narrated notebook: **[`main.ipynb`](main.ipynb)**.

---

## 🔎 Approach

The notebook follows the five required project stages:

### Part 1 — Exploration
Profiling, distributions, and the assignment's seven analysis questions. Key findings:
- The six **expected‑delay** columns are the dominant signal; `expected_dispatch_delay_minutes` alone
  drives arrival lateness (**>60 min departure delay → ~98%** chance of a significant arrival delay).
- **Dispatch hour** rises almost monotonically (~7% at 07:00 → ~21% at 19:00).
- **Provider** and **region/facility** IDs carry real signal (delay rate spans ~8–28%).
- `route_distance_miles` and `internal_operational_score` are **noise** (correlation ≈ 0) and dropped.
- A data‑quality fix: trailing‑whitespace duplicates split provider codes (28 → 15 real codes).

### Part 2 — Preprocessing (leakage‑safe)
A single fitted `ColumnTransformer`, **fit on the training split only** and reused for validation/test:
- **Split first**, then engineer row‑local features (`dispatch_hour`, `arrival_hour`, time‑of‑day in
  minutes, `total_expected_delay`, `n_missing`).
- **Missing values** → median imputation + a missing‑indicator (default), with a switchable
  **KNN** strategy (custom `MedianKNNImputer`, median aggregation).
- **Outliers** → signed‑log transform + optional winsorising (no rows dropped).
- **Scaling** → `StandardScaler`; **categoricals** → one‑hot (low‑cardinality) + target‑encoding
  (high‑cardinality IDs), keeping the matrix compact (~52 columns).

### Part 3 — Models
| Model | Role |
|---|---|
| **Multi‑Layer Perceptron (ANN)** | Primary advanced model — PyTorch, device toggle **CUDA → MPS → CPU**, `BCEWithLogitsLoss` with `pos_weight`, AdamW, dropout/BatchNorm, early stopping |
| **Support Vector Machine** | Linear, **RBF via Nyström approximation**, and a full‑RBF reference on a subsample |
| **Chained MLP → SVM** | Two‑stage: SVM on the MLP's penultimate‑layer embeddings |
| **End‑to‑end MLP + SVM head** | Documented experiment — squared‑hinge (L2‑SVM) loss (Tang, 2013) |
| **Decision Tree** | Advanced model — class‑balanced `DecisionTreeClassifier`, complexity grid + cost‑complexity pruning, interpretable rule splits |
| **Logistic Regression** | Basic linear baseline (`C` tuned by CV) |

> The three required **advanced** models are the **MLP**, the **SVM**, and the **Decision Tree**; the
> chained MLP→SVM and the end‑to‑end SVM head are documented studies under MLP/SVM, and Logistic
> Regression is the basic model.

Both the MLP and the SVMs include **offline grid hyper‑parameter tuning** (median vs. KNN imputation,
architecture/kernel, regularisation), each evaluated by **cross‑validation *and* a held‑out
validation set** with a per‑config confusion matrix. The expensive sweeps are gated behind
`RUN_TUNING` / `RUN_SVM_TUNING` toggles (default **off**); the timed run trains the pre‑selected best.

### Part 4 — Evaluation
Confusion matrix, **Stratified 5‑Fold cross‑validation with combined ROC curves** for every model, and
train‑vs‑validation overfitting checks (preprocessing refit per fold — no leakage). The final
comparison **ranks all models by mean 5‑fold CV AUC** and overlays their cross‑validated ROC curves on
one axes.

### Part 5 — Prediction
Applies the identical fitted pipeline to `test.csv` and writes the submission CSV.

---

## 📊 Results

Models are ranked by **5‑fold cross‑validated AUC** (mean ± std across folds) — more robust than a
single validation split. Everything clusters within **~0.002 AUC**: the engineered delay features make
the problem **near‑saturated** (near‑linearly separable), so the gaps are within fold‑to‑fold noise.

| Rank | Model | 5‑fold CV AUC |
|:--:|---|:--:|
| 🥇 1 | Two‑stage MLP → SVM | **0.9987 ± 0.0003** |
| 2 | **MLP (ANN)** — *final / submission model* | 0.9986 ± 0.0003 |
| 3 | End‑to‑end MLP + SVM head (experiment) | 0.9986 ± 0.0002 |
| 4 | SVM — tuned (KNN + Nyström‑RBF) | 0.9983 ± 0.0003 |
| 5 | Logistic Regression (basic) | 0.9973 ± 0.0004 |
| 6 | Decision Tree (interpretable) | 0.9963 ± 0.0015 |

*Standalone SVM variants explored within the SVM section (held‑out validation AUC, not in the final CV
comparison): Nyström‑RBF ≈ 0.9977 · Linear ≈ 0.9974 · full‑RBF on an 8k subsample ≈ 0.9970.*

**Final choice:** although the two‑stage chain edges ahead **within noise**, the **MLP** is adopted as
the **Part‑5 submission model** — it is the strongest *standalone* model and the simplest to deploy.
The results confirm the data is highly separable: the two‑stage chain shows the MLP embedding is itself
near‑linearly classifiable, the end‑to‑end SVM‑head experiment confirms (per the literature) that
swapping the loss buys nothing here, and the interpretable **Decision Tree** trails only slightly — with
the highest fold‑to‑fold variance, as expected for a single tree.

---

## ▶️ How to run

### 1. Environment
Python **3.12** with the scientific stack (PyTorch for the MLP, scikit‑learn for everything else):

```bash
# conda (recommended)
conda create -n logistics python=3.12 -y
conda activate logistics
pip install "torch>=2.1" "scikit-learn>=1.9" pandas numpy matplotlib seaborn scipy jupyter
```

> The MLP automatically selects the best device: **CUDA → Apple‑Silicon MPS → CPU** (no config needed).

### 2. Data
The course‑provided file names are preserved. By default the notebook reads from `data/`:
- `data/regional_logistics_transfers_train.csv` *(included)*
- `data/regional_logistics_transfers_test.csv` *(add when available — Part 5 then writes the submission)*

To point at another location, set `DATA_DIR` before launching: `export DATA_DIR=/path/to/data`.

### 3. Run the notebook
```bash
jupyter notebook main.ipynb   # then "Run All"
```
Runs **end‑to‑end in ~15–25 minutes** on Apple‑Silicon (MPS) — comfortably within the project's 1‑hour
limit. With a `test.csv` present, Part 5 writes **`Submission_group_15.csv`** (`transfer_id,predict_prob`).

### Reproducing the hyper‑parameter searches (optional)
The full grid searches are slow and meant to run **offline**. To reproduce them, set `RUN_TUNING = True`
(MLP) and/or `RUN_SVM_TUNING = True` (SVM/chained) in their cells and run — otherwise the notebook uses
the already‑selected best configurations.

---

## 🗂️ Repository structure

```text
.
├── main.ipynb                  # ← the full project: Parts 1–5, all models, narrative & plots
├── README.md
├── SOURCES.md                  # research, papers & library references (see below)
├── INSTRUCTIONS.md             # canonical project guidance
├── AGENTS.md                   # entry point for LLM/agent tooling
├── data/
│   └── regional_logistics_transfers_train.csv
├── docs/
│   ├── assignment/             # brief, feature descriptions, submission example
│   ├── course_materials/
│   └── research_materials/     # SVM-architecture literature review
├── reports/
│   └── Report.md               # link to the written report
└── submissions/
    └── final/                  # final deliverables (notebook, report PDF, prediction CSV)
```

---

## 📝 Notes

- **Reproducibility:** every RNG is seeded; preprocessing and model selection use only training‑fold
  data (validation/test are treated as unseen).
- **Calibration is intentionally skipped** — AUC is rank‑based, so monotonic score transforms don't
  change it.
- Required runnable code lives in `main.ipynb`; the notebook is committed **with its outputs**.
- Models implemented: a **Logistic Regression** basic model plus three advanced models — **MLP**,
  **SVM**, and **Decision Tree** — with the chained MLP→SVM and the end‑to‑end SVM head as documented
  studies.

---

## 📚 Sources & references

All research, papers, and library documentation that informed the design are collected in
**[`SOURCES.md`](SOURCES.md)** — organised by model, with a dedicated section on the **chained
MLP→SVM** (two‑stage **and** end‑to‑end, after Tang 2013) and the full bibliography from the
[SVM‑architecture literature review](docs/research_materials/svm_architecture_research.md).

### Final submission checklist (`submissions/final/`)
- `main.ipynb` · `Submission_group_15.pdf` · `Submission_group_15.csv`

---

<sub>Intro to Machine Learning · 2026 · Group #15 — predicting regional logistics transfer delays.</sub>
