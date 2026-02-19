# State Smoother

## Overview

StateSmoother is a PyTorch-based exponential smoothing / state estimation module for time-series data.
It originated as a core component of [New Eden Analytics](https://github.com/New-Eden-Analytics) but has been used standalone in other forecasting
and anomaly-detection workflows.

## Status
Stable prototype. API is mostly stable; test coverage and docs are limited.
Maintained primarily for my own use and as part of [New Eden Analytics](https://github.com/New-Eden-Analytics).


## About

State Smoother was built to improve on common time-series smoothing workflows by
using parameters that are both expressive and easier to reason about than
traditional moving-average style approaches.

The package is used extensively in New Eden Analytics projects to smooth EVE
Online market history data and recover cleaner price signals.

## Installation

A published PyPI package is planned. For now, install from source:

```bash
pip install .
```

## Usage

A practical walkthrough is available in the notebook:

- [demonstration.ipynb](./demonstration.ipynb)

## Project Docs

- Roadmap: [Roadmap.md](./Roadmap.md)
- Version history: [VersionHistory.md](./VersionHistory.md)

## Contributing

Contributions are welcome.

This project follows:

- [Vincent Driessen's Git Branching Model](https://nvie.com/posts/a-successful-git-branching-model/)
- [Semantic Commit Messages](https://seesparkbox.com/foundry/semantic_commit_messages)

## License

This project is licensed under the [GNU General Public License](./LICENSE).
