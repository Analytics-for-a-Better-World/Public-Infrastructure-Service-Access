# Maintained packages

This directory contains independently installable and releasable libraries
maintained within the Public Infrastructure Service Access repository.

Packages belong here when they provide a reusable API, have their own dependency
boundary and test suite, and can be versioned independently from the main PISA
application. Exploratory analyses and country-specific scripts remain under
`Research-Sandbox` and consume these packages through their public APIs.

## Available packages

- [`abw_maxcover`](abw_maxcover): sparse exact and heuristic algorithms for
  maximum-covering solutions and Pareto frontiers.
