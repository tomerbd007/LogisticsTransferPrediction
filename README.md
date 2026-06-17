# Logistics Transfer Prediction

Binary classification project for predicting whether a regional logistics transfer will be delayed by more than 30 minutes.

## Entry Point

Run `main.ipynb` from the repository root. The notebook contains the required project code and narrative for the course submission.

## Directory Map

```text
.
├── main.ipynb
├── INSTRUCTIONS.md
├── AGENTS.md
├── data/
│   └── regional_logistics_transfers_train.csv
├── docs/
│   ├── assignment/
│   └── course_materials/
├── reports/
│   └── Report.md
└── submissions/
    └── final/
```

## Data

The course-provided file names are preserved. By default the notebook reads:

- `data/regional_logistics_transfers_train.csv`
- `data/regional_logistics_transfers_test.csv` when the test set is available

To use another data location, set the `DATA_DIR` environment variable before running the notebook.

## Collaboration

- Treat `INSTRUCTIONS.md` as the canonical project guidance.
- `AGENTS.md` and `.github/instructions/INSTRUCTIONS.instructions.md` are short entry points for LLM agents and GitHub-aware tools.
- Keep required runnable implementation code inside `main.ipynb`.
- Prefer focused notebook changes, because notebooks are difficult to merge.
- Do not commit caches, checkpoints, virtual environments, or intermediate experiment outputs.

## Final Submission Checklist

Place final deliverables in `submissions/final/` when ready:

- `main.ipynb`
- `Submission_group_15.pdf`
- `Submission_group_15.csv`

The final course zip should include only the required notebook, report PDF, and prediction CSV.
