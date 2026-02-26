module CoreNNLS

using LinearAlgebra

export nnls, nnls!, NNLSWorkspace, NNLSResult

struct NNLSResult{T}
    x::Vector{T}
    status::Symbol
    iterations::Int
    residual_norm::T
    kkt_violation::T
end

function Base.iterate(res::NNLSResult, state=1)
    state == 1 && return (res.status, 2)
    state == 2 && return (res.x, 3)
    state == 3 && return (res.iterations, 4)
    state == 4 && return (res.residual_norm, 5)
    state == 5 && return (res.kkt_violation, 6)
    return nothing
end

include("types.jl")        # NNLSWorkspace
include("contracts.jl")    # validate_nnls_inputs, _run_post_checks
include("householder.jl")  # Householder QR (MIT, adapted from Julia stdlib)
include("algorithm.jl")    # nnls!

function nnls(
    A::AbstractMatrix{T},
    b::AbstractVector{T};
    max_iter::Int = 5 * size(A, 2),
) where {T<:AbstractFloat}
    size(A, 1) == length(b) || throw(DimensionMismatch(
        "A hat $(size(A,1)) Zeilen, b hat Länge $(length(b))"))
    any(!isfinite, A) && throw(ArgumentError("NaN/Inf in A"))
    any(!isfinite, b) && throw(ArgumentError("NaN/Inf in b"))

    ws = NNLSWorkspace(size(A, 1), size(A, 2), T; max_iter=max_iter)
    _, x = nnls!(ws, A, b)
    return copy(x)
end

end