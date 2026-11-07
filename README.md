# Modified scikit-learn: Parameterized Impurity Function (PF)

Research source snapshot containing a modified scikit-learn implementation for decision trees and random forests.

## Research modifications
- `sklearn/tree/_criterion.pyx`: `EntropyAlpha`, `GiniBeta`, and `ProposedImpurity`.
- `sklearn/tree/_classes.py`: registers criteria and alpha, beta, gamma parameters.
- `sklearn/ensemble/_forest.py`: exposes and forwards impurity parameters for random forests.

PF: `(H**alpha + G**beta)**gamma`; supported parameter bounds in the supplied source: `0 < alpha <= 1`, `0 < beta <= 0.5`, `0 < gamma <= 1`.

## Installation
This is a modified **source distribution**, not a standard pip release. Follow the included scikit-learn build instructions and use an isolated environment. From `modified-scikit-learn/`, try:

```bash
python -m pip install --editable . --verbose --no-build-isolation --config-settings editable-verbose=true
```

Install the matching build dependencies first. This snapshot has **not** been independently compiled or tested in this review.

## Reproducibility
The experiment scripts, dataset versions, environment versions, and random seeds must be published separately or added to this repository before claiming end-to-end reproducibility. The reported nested-CV protocol must be checked against the experiment scripts.

## License
See `modified-scikit-learn/COPYING` for the upstream scikit-learn license. Retain upstream notices and disclose research modifications.
