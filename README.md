# dune-moes

## Description

The Multithreaded Optimized EigenSolver (MOES) library provides eigensolver algorithms capable of being executed in a multithreaded context. Designed to work as part of the dune environment, they are specialized to approximate eigenspaces of different categories of eigenvalue problems.

## Example usage

See `dune/moes/test/minimalExample.cc`

## Features

- Custom internal data structure to facilitate horizontal SIMD-vectorization using Agner Fog's vectorclass library
- Vectorized implementaion of the sparse matrix-vector product
- Vectorized implementaion of the Gram-Schmidt process
- Vectorized back-insertion of the LU-decomposition

## Requirements

- [DUNE](https://dune-project.org/), needs at least dune-common and dune-istl
- [UMFPACK](https://people.engr.tamu.edu/davis/suitesparse.html)
- [ARPACK](https://github.com/opencollab/arpack-ng) (optional, only needed for comparison tests)
