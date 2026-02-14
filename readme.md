# CoreNNLS.jl

**Status:** Phase 1 Release (Pre-Registration / Development)  
**License:** MIT

**CoreNNLS.jl** is a pure-Julia implementation of the **Lawson-Hanson algorithm** for solving Non-Negative Least Squares (NNLS) problems.

This package is the first standalone product of the **SLSQP.jl** reproduction project. It provides a validated, reference-compatible foundation for solving the Quadratic Programming (QP) sub-problems required in Sequential Quadratic Programming.

---

## Why CoreNNLS.jl?

Unlike existing wrappers or direct Fortran ports, CoreNNLS.jl focuses on:

1. **Equivalence Axiom:** Deterministic behavior that matches the mathematical specification of Lawson & Hanson (1974) and established reference implementations (SciPy/NLopt).
2. **Numerical Forensics:** A transparent, idiomatic Julia codebase that allows for deep inspection of the solver state (via `NNLSWorkspace`).
3. **Type Genericity:** Leveraging Julia's multiple dispatch to support `Float32`, `Float64`, and high-precision types like `BigFloat` or `Double64` without binary overhead.

---
## Standalone Rationale: Breaking the Float64 Barrier

While the Julia ecosystem already offers NNLS solutions, most existing packages are either wrappers around legacy C/Fortran code or are internally restricted to `Float64` precision. This creates a significant bottleneck for:

1. **High-Precision Research:** Solving pathologically ill-conditioned problems (e.g., Hilbert or large-scale Vandermonde matrices).
2. **Numerical Forensics:** Verifying if a solver's failure is due to algorithmic issues or pure rounding errors.
3. **Differentiable Programming:** Integrating NNLS sub-problems into Automatic Differentiation (AD) workflows that require generic types (like `DualNumbers`).

**CoreNNLS.jl** was granted standalone status because it bridges this gap. By utilizing Julia's native type-genericity, it allows users to switch from `Float64` to `BigFloat` or `Double64` with a single type-parameter change—a feat impossible for legacy-wrapped solvers. This makes it not just a component of SLSQP.jl, but a vital tool for high-stakes numerical analysis.
---

## Installation

The package is currently in development and not yet registered in the General Registry. Install it directly via the repository URL:

```julia
using Pkg
Pkg.add(url="[https://github.com/UweAlex/CoreNNLS.jl](https://github.com/UweAlex/CoreNNLS.jl)")

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
ws = NNLSWorkspace(3, 2, Float64)

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
* **Verification:** Integrated Design-by-Contract (DbC) layer for KKT-stationarity checks.

---

## Strategic Context

CoreNNLS.jl was developed following the principle: **Reproduction → Stabilization → Integration**.
While originally built for `SLSQP.jl`, its generic implementation allows it to handle extremely ill-conditioned problems using `BigFloat`, where legacy 64-bit wrappers typically diverge.

---

## References

* Lawson, C. L., & Hanson, R. J. (1974). *Solving Least Squares Problems*.
* Part of the [SLSQP.jl](https://www.google.com/search?q=https://github.com/UweAlex/SLSQP.jl) Project.

```
