# CoreNNLS.jl

**Non-Negative Least Squares (NNLS) solver for Julia — fast, accurate, allocation-free.**

[![Julia](https://img.shields.io/badge/Julia-1.9+-purple)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

CoreNNLS implements the Lawson-Hanson active-set algorithm with an optimised
incremental QR factorisation, column pivoting, and self-diagnosing iterative
refinement. It is significantly faster and more accurate than
[NonNegLeastSquares.jl](https://github.com/rdeits/NonNegLeastSquares.jl) on
medium to large problems.

---

## Installation

```julia
using Pkg
Pkg.add("CoreNNLS")
```

---

## Quick start

```julia
using CoreNNLS

A = randn(1000, 50)
b = randn(1000)

x = nnls(A, b)          # returns Vector{Float64}, all entries ≥ 0
```

Reuse a pre-allocated workspace for repeated solves of the same dimensions
(zero allocations after the first call):

```julia
ws = NNLSWorkspace(1000, 50)

for b in many_right_hand_sides
    _, x = nnls!(ws, A, b)   # allocation-free hot path
end
```

---

## Performance

Benchmarked against NonNegLeastSquares.jl on 126 problems spanning random,
sparse, Toeplitz, Hilbert, and Vandermonde matrices
(3 seeds per configuration, problem set stored in `benchmark/problems.jld2`).

### Speedup by problem size (median over random/sparse/Toeplitz)

| m (rows) | Median speedup |
|---------:|---------------:|
|       50 |          0.56× |
|      200 |          1.26× |
|     1000 |          1.63× |
|     2000 |          1.85× |

CoreNNLS is faster for m ≥ 200. The overhead of incremental QR and column
pivoting does not pay off at m = 50; at m = 2000 the BLAS-accelerated
Householder factorisation dominates.

### Accuracy

Median accuracy ratio (‖Δx‖_NonNeg / ‖Δx‖_Core, higher = CoreNNLS more
accurate): **2.45×** across the full benchmark set.

On sparse problems the advantage is larger: accuracy ratios of 10×–400× are
common because the self-diagnosing KKT refinement detects and corrects
ill-conditioned triangular solves that the competitor silently accepts.

### Selected results

| Problem type | m × n | Speedup | Accuracy |
|:-------------|------:|--------:|---------:|
| Sparse | 1000 × 200 | 3.5–3.8× | 12–15× |
| Sparse | 2000 × 50  | 2.9–4.9× | 100–420× |
| Random | 2000 × 50  | 1.6×     | 46–75×  |
| Toeplitz | 2000 × 200 | 1.6×   | 2×      |
| Hilbert  | 20 × 20    | 1.1×   | equal   |

---

## Features

**Correctness across precisions**

CoreNNLS is generic over `AbstractFloat`. All three precisions produce correct
results:

```julia
x32  = nnls(Float32.(A), Float32.(b))   # Float32, ‖r‖_rel ~ 5e-8
x64  = nnls(A, b)                       # Float64
xbig = nnls(BigFloat.(A), BigFloat.(b)) # arbitrary precision
```

Float32 relative residuals are ≤ 8e-8 on all benchmark problems (well within
single-precision machine epsilon).

**Self-diagnosing iterative refinement (KKT forensic layer)**

After each triangular solve the KKT gradient on the passive set is checked
against a noise floor scaled by `eps(T) · ‖A‖ · ‖x‖`. If the gradient
exceeds this threshold — a sign that the solve accumulated significant
rounding error — one step of iterative refinement is applied at O(m · n_p)
cost. On well-conditioned problems the check never triggers and costs a single
loop over passive variables.

**Incremental QR with column pivoting**

When a variable enters the passive set the existing QR factorisation is
extended by one column in O(m · n_p) rather than rebuilt from scratch.
Rebuilds (after variable removal) use column-pivoted Householder QR for
numerical stability. For non-BLAS types (BigFloat) unpivoted QR is selected
automatically — no configuration needed.

**Zero-allocation hot path**

All buffers are allocated once in `NNLSWorkspace`. The `nnls!` solve path
performs no heap allocation after workspace construction.

---

## API

```julia
# One-shot solve (allocates result)
x = nnls(A, b)
x = nnls(A, b; max_iter = 500)

# In-place solve with pre-allocated workspace
ws = NNLSWorkspace(m, n)               # Float64 by default
ws = NNLSWorkspace(m, n, Float32)      # explicit precision
ws = NNLSWorkspace(m, n; max_iter=500, check_contracts=true)

result = nnls!(ws, A, b)
```

`NNLSResult` fields:

| Field | Description |
|:------|:------------|
| `x` | Solution vector (≥ 0) |
| `status` | `:Success` or `:MaxIterations` |
| `iterations` | Number of outer active-set iterations |
| `residual_norm` | ‖b − Ax‖ |
| `kkt_violation` | max\|wⱼ\| over active set (0 = KKT satisfied) |

---

## When to use CoreNNLS

**Best fit:**
- m ≥ 200 (overdetermined systems, many more rows than columns)
- Accuracy is important — regression, mixture models, spectral unmixing
- Repeated solves of the same dimensions (reuse `NNLSWorkspace`)
- Non-Float64 precision: Float32 for memory-constrained applications,
  BigFloat for reference solutions or ill-conditioned problems

**When NonNegLeastSquares.jl may be faster:**
- Very small problems (m < 50) — incremental QR overhead dominates
- Square or nearly-square problems (m ≈ n)

---

## Reproducing the benchmark

```julia
# Generate problem set once (~10 min for BigFloat(256) reference solutions)
julia --project=. benchmark/generate_problems.jl

# Run full benchmark
julia --project=. Benchmark.jl
```

This generates 132 problems locally (Float64, Float32, and BigFloat(256)
reference solutions for ill-conditioned cases). The generated `problems.jld2`
is not part of the repository — it is created on demand.

---

## Algorithm

CoreNNLS implements the Lawson-Hanson (1974) active-set algorithm with the
following extensions:

- **Incremental column-append QR** — O(m · n_p) column extension avoids full
  rebuilds when variables enter the passive set
- **Column-pivoted Householder QR** on rebuild — improves stability for
  ill-conditioned passive submatrices
- **Diagonal preconditioning** in the triangular solve — reduces the effective
  condition number of R before back-substitution
- **KKT gradient forensic check** — detects and corrects inaccurate solves
  without affecting performance on well-conditioned problems

Reference: Lawson, C. L. & Hanson, R. J. (1974). *Solving Least Squares
Problems*. Prentice-Hall. Chapter 23.

---

## License

MIT — see [LICENSE](LICENSE).