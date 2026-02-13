# src/types.jl
using LinearAlgebra

"""
    NNLSOptions{T}

Parameters controlling solver behavior. Defaults aligned with Phase 0.2.
"""
Base.@kwdef struct NNLSOptions{T}
    w_tol::T = 1e-8            # Dual feasibility tolerance
    max_iter::Int = 1000       # Safety break (anti-cycling)
    rank_tol::T = 1e-10        # Tolerance for detecting singular R diag
end

"""
    NNLSWorkspace{T}

Pre-allocated workspace for Lawson-Hanson NNLS.
Ensures zero allocations during the solve loop.
"""
mutable struct NNLSWorkspace{T}
    # Dimensions
    m::Int
    n::Int

    # Solution & Vectors
    x::Vector{T}        # Solution (size n)
    w::Vector{T}        # Dual vector (gradient), size n
    s::Vector{T}        # Buffer for solution steps (size n)
    r::Vector{T}        # Residual vector r = b - Ax (size m)

    # Active Set Management
    # true = Passive Set P (candidates for non-zero)
    # false = Active Set Z (forced to zero)
    passive_set::BitVector 
    
    # Internal Buffers for QR Decomposition
    # We need a buffer to hold the sub-matrix A[:, P] for the current Passive Set.
    # Allocating this once avoids dynamic allocation in the loop.
    A_passive::Matrix{T} 
    
    # Configuration
    options::NNLSOptions{T}
    
    # State / Diagnostics
    iter::Int
end

# Constructor
function NNLSWorkspace(m::Int, n::Int, T::Type=Float64; kwargs...)
    return NNLSWorkspace{T}(
        m, n,
        zeros(T, n),       # x
        zeros(T, n),       # w
        zeros(T, n),       # s
        zeros(T, m),       # r
        falses(n),         # passive_set (Start: all in Z)
        zeros(T, m, n),    # A_passive buffer (Max size m x n)
        NNLSOptions{T}(;kwargs...),
        0
    )
end
