# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps.

## Types of Contributions

### Reporting Bugs and Feature Requests

Please open bug reports and feature requests on Github.

### Fix Bugs

Fork the repo and when done open a pull request. Two code review
are required to merge into main.

### Write Documentation

You can never have enough documentation! Please feel free to contribute to any
part of the documentation, such as the official docs, docstrings, or the README file.

#### Build Documentation Locally

The documentation is automatically built in Sphinx when the main branch changes. 
It is often useful to build the documentation locally to see how changes to the documentation or the docstrings will be parsed by Sphinx. To do this, activate your virtual environment and in the terminal run `cd docs` and `make html`. This will create a new folder `docs/_build/html`. Click on `index.html` to show the (local) website.

In some cases it can be useful to force Sphinx to rebuild (rather than update) the documentation. To do this follow the steps above but replace `make html` with `make html clean`.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. Try to do many small and focussed pull requests rather than one big one. It makes reviews easier.
2. Make a meaningful description of what the pull request does.
3. If the pull request adds functionality, the docs should be updated.

### Commit Guidelines

We label commits using the *semantic commit messages* [see more](https://www.conventionalcommits.org/en/v1.0.0/).
At the minimum, label your commit with fix, feat, refactor, style or docs and indicate breaking changes with "!".
