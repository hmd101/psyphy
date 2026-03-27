
<div align="center">
    <picture>
      <source srcset="docs/images/psyphy_logo.png" media="(prefers-color-scheme: light)"/>
      <source srcset="docs/images/psyphy_logo.png"  media="(prefers-color-scheme: dark)"/>
      <img alt="psyphy logo" src="docs/images/psyphy_logo.png" width="200">
    </picture>
    <h3>Active-learning-driven adaptive experimentation in psychophysics</h3>
</div>


<h4 align="center">
  <a href="https://flatironinstitute.github.io/psyphy/#install/">Installation</a> |
  <a href="https://flatironinstitute.github.io/psyphy/reference/">Documentation</a> |
  <a href="https://flatironinstitute.github.io/psyphy/examples/">Examples</a> |
  <a href="https://flatironinstitute.github.io/psyphy/CONTRIBUTING/">Contributing</a>
</h4>

## Overview

`psyphy` is an open-source, JAX-based framework for psychophysics research. It leverages GPU acceleration and efficient approximate inference to power Bayesian, active-learning-driven adaptive experiments.

Designed to be modular and extensible, `psyphy` accelerates research workflows and enables real-time adaptive experiments. While currently focused on human color perception, it can be adapted to other perceptual modalities.

The package is under active development, and we welcome contributions.

## Install
`psyphy` only supports python 3.10+. We recommend installing `psyphy` under a virtual environment. Once you've created a virtual environment for `psyphy` and activated it, you can install `psyphy` using pip:

```bash
pip install psyphy
```

If you're developer or want to use the latest features, you can install from GitHub using:
```bash
git clone https://flatironinstitute.github.io/psyphy.git
cd psyphy
pip install -e .

```

## [Quickstart](https://flatironinstitute.github.io/psyphy/examples/wppm/quick_start/)
- Go [here](https://flatironinstitute.github.io/psyphy/examples/wppm/quick_start/) for a light-weight tutorial that demonstrates how to instantiate, evaluate and fit a model quickly. You should be able to run the underlying [script](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/quick_start.py) on your CPU.
- Go [here](https://flatironinstitute.github.io/psyphy/examples/wppm/full_wppm_fit_example/) for a more comprehensive example visualizing a spatially varying covariance field, also explaining the underlying math. The underlying [script](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/full_wppm_fit_example.py) for this tutorial requires a GPU.


## Contributing

For contributors, see  [`CONTRIBUTING.md`](CONTRIBUTING.md) for full doc guidelines and NumPy-style docstrings.
