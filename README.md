# Parameterized Impurity Function for Decision Trees and Random Forests

**Research code and benchmark datasets** for *A Novel Parameterized Impurity Function for Enhancing Decision Tree Construction and Classification Performance*.

This repository contains a modified scikit-learn source tree, ten experiment scripts comparing five splitting criteria across Decision Tree and Random Forest classifiers, and five benchmark CSV datasets. It is intended to support inspection and reproduction of the associated research; successful end-to-end reproduction has **not yet been independently verified**.

## Proposed impurity criterion

For a node containing dataset \(D\), the proposed Parameterized Impurity Function (PF) is

\[
\mathrm{PF}(D;\alpha,\beta,\gamma)=\left[H(D)^\alpha+G(D)^\beta\right]^\gamma,
\]

where \(H(D)\) is Shannon entropy and \(G(D)\) is Gini impurity. The manuscript investigates positive parameters, with the stated theoretical domain \(0<\alpha\leq1\), \(0<\beta\leq0.5\), and \(0<\gamma\leq1\). The source code is authoritative for the parameter grids actually evaluated in each experiment.

Five splitting criteria are compared:

| Criterion | Identifier used in scripts |
| --- | --- |
| Standard entropy | `entropy` |
| Parameterized entropy | `entropy_alpha` |
| Standard Gini | `gini` |
| Parameterized Gini | `gini_beta` |
| Proposed PF | `proposed` |

The nonstandard identifiers require the modified scikit-learn implementation supplied here; a standard installation of scikit-learn will not recognize them.

## Repository layout

```text
PF-DecisionTree-RandomForest/
├── modified-scikit-learn/       # Modified scikit-learn source tree
├── experiments/
│   ├── Decision_Tree/          # Five Decision Tree experiment scripts
│   └── Random_Forest/          # Five Random Forest experiment scripts
├── datasets/                   # Five original CSV files and dataset notes
└── README.md                   # This document
```

### Modified scikit-learn

The custom impurity implementation is principally associated with these source files:

- `modified-scikit-learn/sklearn/tree/_criterion.pyx`
- `modified-scikit-learn/sklearn/tree/_classes.py`
- `modified-scikit-learn/sklearn/ensemble/_forest.py`

The modified tree is a source distribution, **not** a drop-in replacement for an installed Python package. Preserve the upstream license and attribution in `modified-scikit-learn/COPYING`.

### Experiment scripts

| Decision Tree | Random Forest |
| --- | --- |
| `1 Entropy_Decision_Tree.py` | `1 Entropy_Random_Forest_tree.py` |
| `2 Entropy_alpha_Decision_Tree.py` | `2 Entropy_alpha_Random_Forest_tree.py` |
| `3 Gini_Decision_Tree.py` | `3 Gini_Random_Forest_tree.py` |
| `4 Gini_beta_Decision_Tree.py` | `4 Gini_beta_Random_Forest_tree.py` |
| `5 Proposed_Decision_Tree.py` | `5 Proposed_Random_Forest_tree.py` |

Each script contains dataset-specific experiment sections and writes spreadsheet output; inspect its configuration and output filenames before running. Some output filenames include `1000Repeats`, which should **not** be treated as proof that 1,000 repeats are executed by the current script.

## Benchmark datasets

| Filename | Rows | Columns (including target) | Target column |
| --- | ---: | ---: | --- |
| `Breast_Cancer.csv` | 4,024 | 16 | `Status` |
| `online_shoppers_intention.csv` | 12,330 | 18 | `Revenue` |
| `spambase.csv` | 4,601 | 58 | `spam` |
| `TUANDROMD.csv` | 4,464 | 242 | `Label` |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | 7,043 | 21 | `Churn` |

The datasets are supplied as received for the experiments. **Before public redistribution**, verify each dataset's provenance, license, and permission to redistribute. Add original source links and attribution to `datasets/README.md` once verified.

## Environment and installation

A working installation of the **modified** scikit-learn package is required. The source tree includes build configuration (`pyproject.toml` and Meson files). Build instructions and dependency versions should be validated against the exact source revision and the original experiment environment before they are treated as reproducible installation instructions.

Other Python packages imported by the experiment scripts include `numpy`, `pandas`, `scipy`, `joblib`, `matplotlib`, and `xlsxwriter`. These are indicative dependencies, not a tested, pinned environment specification.

A recommended next step is to record Python, NumPy, SciPy, scikit-learn, compiler, and build-tool versions, then test a clean installation in an isolated environment. **No successful build or end-to-end execution is claimed here.**

## Running the experiments

The supplied scripts use relative CSV paths such as `pd.read_csv("Breast_Cancer.csv")`. The simplest way to preserve the scripts unchanged is to execute them **from the `datasets/` directory**, with the repository layout above:

```bash
# From the repository root
cd datasets
python "../experiments/Decision_Tree/1 Entropy_Decision_Tree.py"
python "../experiments/Random_Forest/1 Entropy_Random_Forest_tree.py"
```

Run the other scripts analogously by substituting their exact filenames. Output files may be written to the current working directory; check disk space and available CPU resources before starting computationally expensive runs. The examples assume the modified scikit-learn package has already been successfully installed in the active environment.

## Reproducibility and methodological cautions

- **Random seeds:** The provided scripts do not all use the same repeat index range: some use `range(0, 100)` and others use `range(1, 101)`. Both contain 100 iterations but need not yield paired partitions. The original scripts are retained without silently changing these settings.
- **Parameter selection:** The PF scripts require a formal audit of how parameter tuning is separated from reported evaluation folds. Claims of leakage-free nested cross-validation should not be made until that audit is complete.
- **Execution status:** The scripts were inspected for syntax, but a clean build, full execution, and numerical agreement with the manuscript's tables have not yet been established.
- **Results:** Raw experimental output spreadsheets are not included in the materials reviewed for this README. Add them, along with a manifest and exact software versions, when available.
- **Dataset provenance:** Source URLs and redistribution terms have not yet been verified.

## Code availability statement for the manuscript

> The modified scikit-learn source code and the Decision Tree and Random Forest experiment scripts are available at https://github.com/AbedAlsulami/PF-DecisionTree-RandomForest. The repository also contains the benchmark CSV files used in the experiments. Build verification, dataset provenance, and complete end-to-end reproducibility checks are being documented.

For archival reproducibility, cite a specific Git commit or tagged release rather than relying exclusively on the mutable `main` branch.

## Citation and attribution

If using this code, please cite the associated manuscript once its final bibliographic information is available. The modified scikit-learn distribution retains upstream scikit-learn copyright and licensing terms; consult `modified-scikit-learn/COPYING`.

## Status

**Research release — documentation and validation in progress.** This README describes the supplied code and data, not an independently verified reproduction package.
