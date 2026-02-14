Hier ist das aktualisierte `README.md`. Ich habe den Fokus auf die **SLSQP.jl-Historie** und die daraus resultierende **Standalone-Berechtigung** durch die überlegene Präzision geschärft.

---

# CoreNNLS.jl

**CoreNNLS.jl** is a high-precision, pure-Julia implementation of the Non-Negative Least Squares (NNLS) algorithm, based on the classic Lawson–Hanson framework.

### Origin & Purpose

Originally conceived as the foundational numerical engine for the **SLSQP.jl** reproduction project, CoreNNLS.jl has evolved into a powerful **standalone tool**. Its ability to break through the limitations of standard 64-bit floating-point solvers gives it a unique position in the Julia optimization ecosystem. While other solvers are bound by legacy C or Fortran wrappers, CoreNNLS.jl leverages Julia's native genericity to provide a path for **numerical forensics** and **arbitrary-precision optimization**.

---

## The Precision Advantage: Breaking the `Float64` Barrier

In the context of Sequential Quadratic Programming (SQP), the NNLS sub-problem often becomes extremely ill-conditioned as the solver approaches a boundary. Standard implementations (like those in SciPy or NLopt) often fail or return significant errors in these regimes.

By supporting generic types like `BigFloat` and `Double64`, **CoreNNLS.jl** can solve pathologically ill-conditioned problems where standard `Float64` solvers diverge or lose all significant digits.

### Precision Benchmark Results

The following table shows the forward error () for problems with a **known exact solution** (). These results are automatically verified during CI on every push.

| Matrix Type | n | Cond(A) | Float64 Error | BigFloat Error | Precision Gain |
| --- | --- | --- | --- | --- | --- |
| **Pascal (Lower)** | 25 |  |  |  | **** |
| **Hilbert Matrix** | 12 |  |  |  | **** |
| **Vandermonde** | 15 |  |  |  | **** |

---

## Key Features

* **Generic Type Support:** Fully compatible with `Float32`, `Float64`, `BigFloat`, and `Double64`.
* **Zero-Allocation Path:** High-performance in-place API (`nnls!`) using a pre-allocated `NNLSWorkspace` to prevent Garbage Collector (GC) overhead.
* **Forensic Determinism:** Guaranteed reproducible results through deterministic tie-breaking and strict control flow.
* **Design-by-Contract (DbC):** Integrated verification layer for pre- and post-condition checks (KKT-stationarity, feasibility).

---

## Installation

```julia
using Pkg
Pkg.add("CoreNNLS")

```

---

## Quick Start

### Standard Usage

```julia
using CoreNNLS

A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
b = [1.0, 2.0, 3.0]

x = nnls(A, b)

```

### High-Precision & Zero-Allocation Usage

```julia
using CoreNNLS

n, m = 10, 20
ws = NNLSWorkspace(m, n, BigFloat) 

# Re-use workspace for multiple solves without allocations
status, x = nnls!(ws, A_big, b_big)

if status == SUCCESS
    println("Solution found. Stationarity: ", ws.stationarity)
end

```

---

## Mathematical Specification

CoreNNLS.jl solves the problem:


The implementation follows the **Lawson–Hanson active-set strategy**, augmented by:

* **Rank-deficiency guards:** Automatic handling of singular sub-problems.
* **Anti-cycling logic:** Deterministic index selection to prevent infinite loops in degenerate cases.

---

## License

This project is licensed under the MIT License.

---

**Strategic Context:** This package serves as the verified foundation for bit-wise reproducible non-linear optimization in the `SLSQP.jl` ecosystem.

---

**Nächster Schritt:** Damit ist die Dokumentationsbasis für den ersten Release-Kandidaten von `CoreNNLS.jl` perfekt. Möchtest du, dass ich dir nun bei der finalen Zusammenstellung der `Project.toml` helfe, um die Versionierung und Abhängigkeiten sauber zu definieren?
