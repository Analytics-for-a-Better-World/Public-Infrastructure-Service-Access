# Maintained packages

This directory contains independently installable and releasable libraries
maintained within the Public Infrastructure Service Access repository.

Packages belong here when they provide a reusable API, have their own dependency
boundary and test suite, and can be versioned independently from the main PISA
application. Exploratory analyses and country-specific scripts remain under
`Research-Sandbox` and consume these packages through their public APIs.
Install a package to use it, for example `python -m pip install -e
packages/abw_maxcover`. Scripts under `Research-Sandbox` prefer the installed
package and only add the source tree to `sys.path` when it is not installed.

## Available packages

- [`abw_maxcover`](abw_maxcover): sparse exact and heuristic algorithms for
  maximum-covering solutions and Pareto frontiers.
