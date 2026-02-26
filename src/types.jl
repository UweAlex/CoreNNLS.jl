# =============================================================================
# File:    types.jl
# Project: CoreNNLS.jl
# Date:    2025-02
# Purpose: Core data structures for the Lawson-Hanson NNLS solver.
#          Defines solver options, the pre-allocated workspace, and
#          the active-set management routines that operate on it.
#
# Design:  All heap allocations happen once at workspace construction.
#          The hot path (nnls!) is allocation-free after that.
# =============================================================================

# -----------------------------------------------------------------------------
# Solver options
# -----------------------------------------------------------------------------

"""
    NNLSOptions{T<:AbstractFloat}

Immutable configuration for the NNLS solver.

Fields
------
- `max_iter`        : Maximum number of outer (active-set) iterations.
- `check_contracts` : Enable Design-by-Contract post-condition checks (slow).
"""
struct NNLSOptions{T<:AbstractFloat}
    max_iter::Int
    check_contracts::Bool
end

"""
    NNLSOptions{T}(; max_iter=1000, check_contracts=false)

Construct solver options for element type `T`.

# Keyword arguments
- `max_iter`        : Upper bound on outer iterations (default: 1000).
- `check_contracts` : Run KKT and non-negativity checks after solve (default: false).

# Returns
`NNLSOptions{T}`
"""
function NNLSOptions{T}(;
    max_iter::Int         = 1000,
    check_contracts::Bool = false,
) where {T<:AbstractFloat}
    return NNLSOptions{T}(max_iter, check_contracts)
end

# -----------------------------------------------------------------------------
# Pre-allocated workspace
# -----------------------------------------------------------------------------

"""
    NNLSWorkspace{T<:AbstractFloat}

Mutable workspace holding all buffers required by `nnls!`.
Allocate once with `NNLSWorkspace(m, n)`, then reuse across multiple solves
of the same problem dimensions to avoid repeated heap allocation.

Fields
------
Dimensions:
- `m`, `n`           : Problem size (m rows, n columns).

Primal / dual vectors:
- `x`                : Current non-negative solution estimate (length n).
- `w`                : Dual / gradient vector w = Aᵀ(b − Ax) (length n).
- `s`                : Unconstrained least-squares solution on passive set (length n).
- `r`                : Residual r = b − Ax (length m).

Active-set book-keeping:
- `passive_set`      : BitVector — passive_set[j] = true iff variable j is in the passive set.
- `passive_indices`  : Ordered list of passive column indices (1-indexed, length n).
- `n_passive`        : Number of currently passive variables.

QR factorisation buffers:
- `A_passive`        : Compact QR storage — columns of the passive submatrix,
                       overwritten in-place by Householder reflectors (m × n).
- `tau`              : Householder scalars τₖ for each reflector (length n).
- `perm`             : Column permutation for pivoted QR rebuild;
                       perm[k] = position in passive_indices (length n).
- `col_norms_sq`     : Squared column norms used for pivot selection (length n).
- `Qtb_work`         : Scratch buffer for Qᵀb computation — keeps `r` invariant (length m).
- `r_scale`          : Diagonal preconditioning factors for the triangular solve;
                       r_scale[k] = 1/|R[k,k]|.  Computed on-the-fly when the
                       estimated condition number of R exceeds 1/√eps(T).  Length n.

Solver state:
- `options`          : `NNLSOptions{T}` configuration.
- `iter`             : Iteration counter (reset to 0 at the start of each solve).
"""
mutable struct NNLSWorkspace{T<:AbstractFloat}
    m::Int
    n::Int

    x::Vector{T}
    w::Vector{T}
    s::Vector{T}
    r::Vector{T}

    passive_set::BitVector
    passive_indices::Vector{Int}
    n_passive::Int

    A_passive::Matrix{T}
    tau::Vector{T}
    perm::Vector{Int}
    col_norms_sq::Vector{T}
    Qtb_work::Vector{T}
    r_scale::Vector{T}   # diagonal preconditioning factors 1/|R[k,k]| (length n)

    options::NNLSOptions{T}
    iter::Int
end

"""
    NNLSWorkspace(m, n [, T=Float64]; max_iter=1000, check_contracts=false)

Allocate a workspace for an m × n NNLS problem of element type `T`.

# Arguments
- `m`               : Number of rows.
- `n`               : Number of columns.
- `T`               : Element type (default: `Float64`; also supports `Float32`, `BigFloat`).

# Keyword arguments
- `max_iter`        : Maximum outer iterations (default: 1000).
- `check_contracts` : Enable post-solve correctness checks (default: false).

# Returns
`NNLSWorkspace{T}` — ready for use with `nnls!`.

# Example
```julia
ws = NNLSWorkspace(100, 30)
result = nnls!(ws, A, b)
```
"""
function NNLSWorkspace(m::Int, n::Int, ::Type{T}=Float64;
                       max_iter::Int         = 1000,
                       check_contracts::Bool = false) where {T<:AbstractFloat}
    return NNLSWorkspace{T}(
        m, n,
        zeros(T, n),   # x
        zeros(T, n),   # w
        zeros(T, n),   # s
        zeros(T, m),   # r
        falses(n),           # passive_set
        zeros(Int, n),       # passive_indices
        0,                   # n_passive
        zeros(T, m, n),      # A_passive
        zeros(T, n),         # tau
        collect(1:n),        # perm: identity permutation as starting point
        zeros(T, n),         # col_norms_sq
        zeros(T, m),         # Qtb_work
        ones(T, n),          # r_scale: identity until preconditioning is needed
        NNLSOptions{T}(; max_iter=max_iter, check_contracts=check_contracts),
        0,                   # iter
    )
end

# -----------------------------------------------------------------------------
# Active-set management
# -----------------------------------------------------------------------------

"""
    reset_passive!(ws)

Clear the passive set: mark all variables as active and reset the counter.

# Arguments
- `ws` : `NNLSWorkspace` to reset.
"""
function reset_passive!(ws::NNLSWorkspace)
    fill!(ws.passive_set, false)
    ws.n_passive = 0
    return nothing
end

"""
    add_passive!(ws, j)

Move variable `j` from the active set into the passive set.
No-op if `j` is already passive.

# Arguments
- `ws` : `NNLSWorkspace`.
- `j`  : Column index (1-indexed) to add to the passive set.
"""
function add_passive!(ws::NNLSWorkspace, j::Int)
    if !ws.passive_set[j]
        ws.passive_set[j] = true
        ws.n_passive += 1
        ws.passive_indices[ws.n_passive] = j
    end
    return nothing
end

"""
    remove_passive_at!(ws, pos)

Remove the variable at position `pos` in `passive_indices` from the passive set.
Shifts the remaining entries left to keep the list contiguous.

# Arguments
- `ws`  : `NNLSWorkspace`.
- `pos` : Position (1-indexed) within `passive_indices` to remove.
"""
function remove_passive_at!(ws::NNLSWorkspace, pos::Int)
    j = ws.passive_indices[pos]
    ws.passive_set[j] = false
    for k in pos:(ws.n_passive - 1)
        ws.passive_indices[k] = ws.passive_indices[k + 1]
    end
    ws.n_passive -= 1
    return nothing
end