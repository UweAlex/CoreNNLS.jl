#src/CoreNNLS.jl
module CoreNNLS

using LinearAlgebra

# Include components (order matters: types → contracts → algorithm)
include("types.jl")
include("contracts.jl")
include("algorithm.jl")

# Public API
export NNLSWorkspace, NNLSOptions
export nnls, nnls!

"""
    nnls(A, b; kwargs...) -> x

Convenience wrapper: solve  min ||Ax - b||₂  s.t.  x ≥ 0.

All keyword arguments are forwarded to [`NNLSOptions`](@ref).
Warns if the solver does not converge.

See also [`nnls!`](@ref) for the allocation-reusing in-place variant.
"""
function nnls(A::AbstractMatrix{T}, b::AbstractVector{T}; kwargs...) where {T<:AbstractFloat}
    m, n = size(A)
    length(b) == m || throw(DimensionMismatch("A has $m rows but b has length $(length(b))"))
    ws = NNLSWorkspace(m, n, T; kwargs...)
    status, x = nnls!(ws, A, b)
    if status == :MaxIter
        @warn "NNLS did not converge within $(ws.options.max_iter) iterations. " *
              "Result may be suboptimal. Consider increasing max_iter."
    end
    return x
end

end # module CoreNNLS
