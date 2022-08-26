
# Multi-layered neural networks of drifting assemblies
Simon Altrogge

*Neural Network Dynamics and Computation, Institute of Genetics, University of Bonn, Bonn, Germany.*

---

This repository contains the [data](/data) and [code](/src) to reproduce the main results of my master's thesis.
Among others, we provide three Python packages helpful to simulate drifting assemblies in networks of Poisson model
neurons:
- [neuralnetwork](/src/neuralnetwork)
- [normalization](/src/normalization)
- [clusterings](/src/clusterings)

## Requirements

The code was written in Python 3.10.4 using [NumPy](https://numpy.org/) 1.22.3, [SciPy](https://scipy.org/) 1.8.0,
[xarray](https://xarray.dev/) 2022.3.0, [bctpy](https://github.com/aestrivex/bctpy) 0.5.2, and
[Matplotlib](https://matplotlib.org/) 3.5.1. The full conda environment can be reproduced using the
[environment file](/environment.yaml).
