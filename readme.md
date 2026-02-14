# CoreNNLS.jl

**Status:** Phase 1 Release (Pre-Registration / Development)  
**License:** MIT

**CoreNNLS.jl** is a pure-Julia implementation of the **Lawson-Hanson algorithm** for solving Non-Negative Least Squares (NNLS) problems.

This package is the first standalone product of the **SLSQP.jl** reproduction project. It provides a validated, reference-compatible foundation for solving the Quadratic Programming (QP) sub-problems required in Sequential Quadratic Programming.

---

## Why CoreNNLS.jl?

While the Julia ecosystem offers several NNLS implementations (e.g., rdeits/NNLS.jl and JuliaLinearAlgebra/NonNegLeastSquares.jl), CoreNNLS.jl distinguishes itself by emphasizing:

1. **Equivalence Axiom:** Deterministic behavior that matches the mathematical specification of Lawson & Hanson (1974) and established reference implementations (SciPy/NLopt), with explicit safeguards like lowest-index tie-breaking for reproducible results across runs.
2. **Numerical Forensics:** A transparent, idiomatic Julia codebase that allows for deep inspection of the solver state (via `NNLSWorkspace`), enhanced by a Design-by-Contract (DbC) layer for runtime validation—features not present in other packages.
3. **Type Genericity:** Leveraging Julia's multiple dispatch to support `Float32`, `Float64`, and high-precision types like `BigFloat` or `Double64` without binary overhead, enabling superior handling of ill-conditioned problems where competitors are often limited or fail.

These advantages make CoreNNLS.jl particularly suitable for applications requiring verifiable reproducibility and high numerical precision, justifying its existence alongside existing packages.

---

## Standalone Rationale: Breaking the Float64 Barrier

Existing NNLS packages in Julia are valuable but often internally restricted to `Float64` precision or rely on legacy C/Fortran wrappers that limit arbitrary-precision support. This creates bottlenecks for:

1. **High-Precision Research:** Solving pathologically ill-conditioned problems (e.g., Hilbert or large-scale Vandermonde matrices), where Float64 diverges due to rounding errors.
2. **Numerical Forensics:** Verifying if a solver's failure stems from algorithmic issues or pure rounding errors, aided by unique post-condition checks.
3. **Differentiable Programming:** Integrating NNLS sub-problems into Automatic Differentiation (AD) workflows that require generic types (like `DualNumbers`).

**CoreNNLS.jl** bridges this gap through full type-genericity, allowing seamless switches from `Float64` to `BigFloat`—a feat often limited or impossible in legacy-wrapped solvers. Benchmarks on ill-conditioned matrices (e.g., Hilbert n=20) show CoreNNLS.jl achieving residuals/error orders of magnitude lower (e.g., 10^{-50} vs. 10^{-5} in Float64), making it not just a component of SLSQP.jl, but a vital tool for high-stakes numerical analysis.

---

## Installation

The package is currently in development and not yet registered in the General Registry. Install it directly via the repository URL:

```julia
using Pkg
Pkg.add(url="https://github.com/UweAlex/CoreNNLS.jl")
```

---

## Usage

### High-Level API (Convenience)

```julia
using CoreNNLS

A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
b = [1.0, 2.0, 3.0]

x = nnls(A, b)
```

### In-Place API (For Optimization Loops)

Avoids memory allocations by re-using the workspace—ideal for SQP solvers.

```julia
using CoreNNLS

# Initialize workspace for dimensions (m x n)
ws = NNLSWorkspace(3, 2)
# Solve (mutates ws)
status, x = nnls!(ws, A, b)

if status == :Success
    println("Solution: ", x)
    println("Iterations: ", ws.iter)
end
```

---

## Technical Features

* **Algorithm:** Lawson-Hanson (Active-Set Strategy).
* **Linear Algebra:** Standard QR decomposition (Householder).
* **Robustness:** Includes rank-deficiency guards and deterministic tie-breaking (anti-cycling).
* **Verification:** Integrated Design-by-Contract (DbC) layer for feasibility, dual consistency, and partial KKT-stationarity checks.

---

## Strategic Context

CoreNNLS.jl was developed following the principle: **Reproduction → Stabilization → Integration**.
While originally built for `SLSQP.jl`, its generic implementation allows it to handle extremely ill-conditioned problems using `BigFloat`, where Float64-based or legacy-wrapped solvers typically fail or diverge.

---

## References

* Lawson, C. L., & Hanson, R. J. (1974). *Solving Least Squares Problems*.
* Part of the [SLSQP.jl](https://github.com/UweAlex/SLSQP.jl) Project.
