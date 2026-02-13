# src/CoreNNLS.jl
module CoreNNLS

using LinearAlgebra

# Include components
include("types.jl")
include("contracts.jl")
include("algorithm.jl")

# Export API
export NNLSWorkspace, NNLSOptions
export nnls, nnls!

# Convenience Out-of-Place Wrapper
function nnls(A::AbstractMatrix{T}, b::AbstractVector{T}; kwargs...) where {T}
    m, n = size(A)
    ws = NNLSWorkspace(m, n, T; kwargs...)
    status, x = nnls!(ws, A, b)
    return x
end

end # module
