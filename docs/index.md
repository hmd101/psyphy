<div align="center">
    <picture>
      <source srcset="images/psyphy_logo.png" media="(prefers-color-scheme: light)"/>
      <source srcset="images/psyphy_logo.png"  media="(prefers-color-scheme: dark)"/>
      <img alt="psyphy logo" src="images/psyphy_logo.png" width="200">
    </picture>
    <h3>Active-learning-driven adaptive experimentation in psychophysics</h3>
</div>




<h4 align="center">
  <a href="https://flatironinstitute.github.io/psyphy/reference/">Documentation</a> |
  <a href="https://flatironinstitute.github.io/psyphy/examples/">Examples</a> |
  <a href="https://flatironinstitute.github.io/psyphy/CONTRIBUTING/">Contributing</a>
</h4>



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

## What is in psyphy?

`psyphy` is an open-source, JAX-based framework for psychophysics research. It leverages GPU acceleration and efficient approximate inference to power Bayesian, active-learning-driven adaptive experiments.

Designed to be modular and extensible, `psyphy` accelerates research workflows and enables real-time adaptive experiments. While currently focused on human color perception, it can be adapted to other perceptual modalities.
