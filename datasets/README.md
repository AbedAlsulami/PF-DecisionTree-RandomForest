# Experimental datasets

This folder contains five CSV datasets supplied by the study author for the Decision Tree and Random Forest experiments. The original CSV files are included without modification.

| File | Rows | Columns (including target) | Target |
|---|---:|---:|---|
| `Breast_Cancer.csv` | 4,024 | 16 | `Status` |
| `online_shoppers_intention.csv` | 12,330 | 18 | `Revenue` |
| `spambase.csv` | 4,601 | 58 | `spam` |
| `TUANDROMD.csv` | 4,464 | 242 | `Label` |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | 7,043 | 21 | `Churn` |

## Usage

Place this `datasets/` folder at the repository root, alongside `experiments/` and `modified-scikit-learn/`. Update dataset paths in experiment scripts if their current paths differ; do not change the CSV contents.

Example Python loading from the repository root:

```python
from pathlib import Path
import pandas as pd

dataset_path = Path("datasets") / "Breast_Cancer.csv"
df = pd.read_csv(dataset_path)
```

## Provenance and redistribution

These CSV files were supplied by the study author. Their original publication sources, versions, and redistribution licenses have not yet been independently verified. Before publishing this folder in a public GitHub repository, verify that each dataset may legally be redistributed, document its original source and license, and review it for any sensitive information. If redistribution is not permitted, replace the file with a source/download link and instructions instead.

## Reproducibility

Publishing the datasets alone does not establish reproducibility. Document the precise package versions, modified scikit-learn build instructions, experiment commands, seeds, and validation protocol in the repository README.
