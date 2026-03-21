![psyphy logo](docs/images/psyphy_logo_draft.png)

<div align="center">
    <picture>
    <source srcset="docs/images/psyphy_logo_draft.png" media="(prefers-color-scheme: light)"/>
    <source srcset="docs/images/psyphy_logo_draft.png"  media="(prefers-color-scheme: dark)"/>
    <!-- <img align="center" src="docs/assets/logo/logo_text_black.svg" alt="Inferno" width="400" style="padding-right: 10px; padding left: 10px;"/> -->
    </picture>
    <h3>Psychophysical Modeling and Adaptive Trial Placement</h3>
</div>


<h4 align="center">
  <a href="https://flatironinstitute.github.io/psyphy/#install/">Installation</a> |
  <a href="https://flatironinstitute.github.io/psyphy/reference/">Documentation</a> |
  <a href="https://flatironinstitute.github.io/psyphy/examples/">Examples</a> |
  <a href="https://flatironinstitute.github.io/psyphy/CONTRIBUTING/">Contributing</a>
</h4>


## Install (editable)

```bash
git clone https://flatironinstitute.github.io/psyphy.git
cd psyphy
pip install -e .

```

## [Quickstart](https://flatironinstitute.github.io/psyphy/examples/wppm/quick_start/)
[Go here](https://flatironinstitute.github.io/psyphy/examples/wppm/quick_start/) for a light-weight tutorial that demonstrates how to instantiate, evaluate and fit a model quickly. 


## This package provides:

- Wishart Process Psychophysical Model (WPPM)
    - fit to subject's data
    - predict psychphysical thresholds
    - optional trial placement strategy leveraging model's posterior (e.g., information gain, place next batch of trials such that model's uncertainty is maximally reduced)
- Priors and noise models
    - supports cold and warm starts where warm means initialzing with parameters from previous subjects fitted parameters
    - Noise Model:
        - default: Gaussian
        - supports Student's T
- Task likelihoods
    - currently supports OddityTask, TwoAFC
- Inference engines (MAP, Langevin, Laplace)
- Posterior wrappers and diagnostics
- Trial placement strategies (grid, information gain)
    - supports online and batchwise trial placement
- Experiment session orchestration
    - reading session data and exporting next batch of trial placments



## Background

This package implements methods described in:
-  [Hong et al. (2025). *Comprehensive characterization of human color discrimination thresholds*.](https://www.biorxiv.org/content/10.1101/2025.07.16.665219v1)

While the paper above  used AEPsych (a Gaussian Process–based trial placer),
`psyphy` integrates trial placement directly with the WPPM posterior (e.g. via InfoGain/EAVC),
making the  adaptive trial placement model-aware.

## Docs

Build and preview the documentation locally:

```bash
# from repo root
source .venv/bin/activate
pip install mkdocs mkdocs-material 'mkdocstrings[python]'
mkdocs serve
```

Build the static site:

```bash
mkdocs build
```

Deploy to GitHub Pages (manual):

```bash
mkdocs gh-deploy --clean
```

For contributors, see CONTRIBUTING.md for full doc guidelines and NumPy-style docstrings.
