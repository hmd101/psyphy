# Contributing to psyphy

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

We actively welcome your pull requests.

1.  Fork the repo and create your branch from `main`.
2.  If you've added code that should be tested, add tests.
3.  If you've changed APIs, update the documentation.
4.  Ensure the test suite passes.
5.  Make sure your code lints. See [LINTING_SETUP.md](LINTING_SETUP.md) for details.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to `psyphy`, you agree that your contributions will be licensed
under the [`LICENSE.md`](LICENSE.md) file in the root directory of this source tree.


## Docs

Build and preview the documentation locally:

```bash
# from repo root
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