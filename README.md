# State Smoother

`StateSmoother` is a PyTorch-based time-series state smoothing library for tabular observations.

## Overview
The package exposes a `Smoother` class for fitting smoothed state trajectories from a pandas `DataFrame`. It supports multi-column series, derivative dimensions, configurable smoothing horizon, and pluggable error functions.

## Installation
Install from source:

```bash
pip install .
```

## Python Version
- `Python >= 3.11`

## Support
- Supported Python: `3.11+`
- Distribution status: source install from this repository

## Quick Start
```python
import pandas as pd

from state_smoother import Smoother

data = pd.DataFrame(
    {
        "alpha": [10.0, 11.0, 13.0, 12.0, 14.0, 15.0],
        "beta": [2.0, 3.0, 2.0, 4.0, 3.0, 5.0],
    }
)

model = Smoother(data, dims=3, horizon=4)
model.learn(epocs=10, learn_rate=1e-2)

smoothed = model.state
loss_history = model.loss_rcd
```

For a longer walkthrough, see [`demonstration.ipynb`](./demonstration.ipynb).

## Implemented API
### Constructor
- `Smoother(data_frame, dims, horizon, error_func=sq_err, seed_data=None, coef_targets=0, learn_seed=False, alpha=0.2, beta=0.1, learn_clamp=1e-16, flow_clamp=32, verbose=False, tqdm_leave=True)`

### Main Attributes
- `coefs`
- `loss_rcd`
- `actuals`
- `state`

### Methods
- `learn(epocs, learn_rate=1e-3)`
- `plot_fit(col, dim=0, figsize=(14, 10), ax=None)`

### Error Functions
- `sq_err(prediction, actual)`
- `log_perc_error(prediction, actual)`

## Input Expectations
The current implementation expects:
- `data_frame` as a pandas `DataFrame`
- `dims >= 1`
- `horizon > 0`
- numeric input values compatible with PyTorch tensor conversion

Behavior to be aware of:
- If `seed_data` is omitted, the first row of `data_frame` is used as the seed and removed from the learned observations.
- If one-dimensional `seed_data` is provided, higher derivative dimensions are zero-filled.
- Setting `learn_seed=True` adds the seed to the learned coefficients, but the implementation notes this is less stable and slower to train.

## Testing
The project includes a pytest suite under [`tests/`](./tests/), including coverage for:
- learning output shapes and loss history
- default and explicit seed behavior
- flow initialization and smoothing helpers
- built-in error functions
- smoke coverage against sample EVE market data

Run tests with:
```bash
uv run pytest -q
```

## Development Setup
Install development dependencies:
```bash
uv sync --dev
```

Run tests:
```bash
uv run pytest -q
```

## Version History
See [`VersionHistory.md`](./VersionHistory.md).

## Contributing
- Contributions are welcome.
- This project follows [GitFlow](https://nvie.com/posts/a-successful-git-branching-model/).
- Semantic commit messages are encouraged.

## AI Usage
- AI tooling may assist with test authoring, documentation drafting/editing, and development guidance for planning and decision support.
- Do not treat AI-generated content as authoritative by default; repository code and maintained documentation remain the source of truth.

## License
This project is licensed under the GNU GPL. See [`LICENSE`](./LICENSE).
