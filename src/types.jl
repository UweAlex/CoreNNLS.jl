# src/types.jl
"""
    NNLSOptions{T}

Parameters controlling solver behavior.

# Fields
- `dual_tol`:  Tolerance for dual feasibility (KKT optimality of free gradient).
- `feas_tol`:  Tolerance for feasibility of intermediate LS solutions (s >= -feas_tol).
- `zero_tol`:  Threshold below which a variable is clamped to zero.
- `rank_tol`:  Tolerance for detecting rank deficiency in QR diagonal.
- `max_iter`:  Maximum outer iterations (anti-cycling safeguard).
- `check_contracts`: Enable runtime DbC post-condition checks (default: false).
"""
Base.@kwdef struct NNLSOptions{T<:AbstractFloat}
    dual_tol::T        = T(100) * eps(T)   # Dual feasibility / optimality
    feas_tol::T        = T(100) * eps(T)   # LS sub-problem feasibility
    zero_tol::T        = T(100) * eps(T)   # Zero-clamping threshold
    rank_tol::T        = T(10)  * eps(T)   # QR rank detection
    max_iter::Int      = 1000              # Safety break
    check_contracts::Bool = false           # Enable DbC post-condition checks
end

"""
    NNLSWorkspace{T}

Pre-allocated workspace for Lawson-Hanson NNLS.

Maintains a passive index list for O(n_p) inner-loop operations
instead of O(n) full scans.
"""
mutable struct NNLSWorkspace{T<:AbstractFloat}
    # Dimensions
    m::Int
    n::Int

    # Solution & Vectors
    x::Vector{T}           # Solution (size n)
    w::Vector{T}           # Dual vector (gradient), size n
    s::Vector{T}           # Buffer for LS sub-problem solution (size n)
    r::Vector{T}           # Residual vector r = b - Ax (size m)

    # Active Set Management
    passive_set::BitVector          # true = Passive (candidate for non-zero)
    passive_indices::Vector{Int}    # Ordered list of passive column indices
    n_passive::Int                  # Current count of passive variables

    # Internal Buffers for QR Decomposition
    A_passive::Matrix{T}           # Buffer for sub-matrix A[:, P] (max size m × n)

    # Configuration
    options::NNLSOptions{T}

    # State / Diagnostics
    iter::Int
end

"""
    NNLSWorkspace(m, n, [T=Float64]; kwargs...)

Construct workspace for an m×n NNLS problem with element type `T`.
All keyword arguments are forwarded to [`NNLSOptions`](@ref).
"""
function NNLSWorkspace(m::Int, n::Int, ::Type{T}=Float64; kwargs...) where {T<:AbstractFloat}
    return NNLSWorkspace{T}(
        m, n,
        zeros(T, n),        # x
        zeros(T, n),        # w
        zeros(T, n),        # s
        zeros(T, m),        # r
        falses(n),          # passive_set
        zeros(Int, n),      # passive_indices
        0,                  # n_passive
        zeros(T, m, n),     # A_passive buffer
        NNLSOptions{T}(; kwargs...),
        0,
    )
end

# --------------------------------------------------------------------------
# Passive index list helpers
# --------------------------------------------------------------------------

"""Reset the passive set and index list to empty (all variables active/zero)."""
function reset_passive!(ws::NNLSWorkspace)
    fill!(ws.passive_set, false)
    ws.n_passive = 0
end

"""Add column `j` to the passive set."""
function add_passive!(ws::NNLSWorkspace, j::Int)
    if !ws.passive_set[j]
        ws.passive_set[j] = true
        ws.n_passive += 1
        ws.passive_indices[ws.n_passive] = j
        # keep sorted for deterministic QR column order
        sort!(@view ws.passive_indices[1:ws.n_passive])
    end
end

"""Remove column `j` from the passive set."""
function remove_passive!(ws::NNLSWorkspace, j::Int)
    if ws.passive_set[j]
        ws.passive_set[j] = false
        # compact the index list
        write_pos = 0
        for k in 1:ws.n_passive
            if ws.passive_indices[k] != j
                write_pos += 1
                ws.passive_indices[write_pos] = ws.passive_indices[k]
            end
        end
        ws.n_passive = write_pos
    end
end
end
