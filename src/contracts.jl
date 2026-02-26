function validate_nnls_inputs(
    A::AbstractMatrix{T}, b::AbstractVector{T}, ws::NNLSWorkspace{T}
) where {T}
    size(A) == (ws.m, ws.n) || throw(
        DimensionMismatch("A is $(size(A)), workspace expects ($(ws.m), $(ws.n))")
    )
    length(b) == ws.m || throw(
        DimensionMismatch("b has length $(length(b)), expected $(ws.m)")
    )
    any(!isfinite, A) && throw(ArgumentError("A contains NaN or Inf entries"))
    any(!isfinite, b) && throw(ArgumentError("b contains NaN or Inf entries"))
    return nothing
end

function validate_nnls_post(
    A::AbstractMatrix{T}, b::AbstractVector{T},
    x::AbstractVector{T}, w::AbstractVector{T},
    tol::T
) where {T}
    ok = true
    n = length(x)

    if minimum(x) < -tol
        @warn "NNLS post-condition: x negative"
        ok = false
    end

    r = b - A * x
    w_calc = A' * r
    for j in 1:n
        if x[j] > tol && abs(w_calc[j]) > tol * T(100)
            @warn "NNLS post-condition: complementarity at j=$j"
            ok = false
            break
        end
    end

    return ok
end

function _run_post_checks(ws::NNLSWorkspace{T}, A, b) where {T}
    if ws.options.check_contracts
        max_abs_A = zero(T)
        @inbounds for j in 1:ws.n, i in 1:ws.m
            v = abs(A[i, j])
            if v > max_abs_A; max_abs_A = v; end
        end
        tol = eps(T) * max_abs_A * T(max(ws.m, ws.n))
        validate_nnls_post(A, b, ws.x, ws.w, tol)
    end
end