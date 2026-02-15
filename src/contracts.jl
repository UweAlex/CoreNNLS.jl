"""
    validate_nnls_inputs(A, b, ws)

Pre-condition checks: dimensions, finiteness.
Throws on violation.
"""
function validate_nnls_inputs(
    A::AbstractMatrix{T}, b::AbstractVector{T}, ws::NNLSWorkspace{T}
) where {T}
    size(A) == (ws.m, ws.n) || throw(
        DimensionMismatch(
            "A is $(size(A)), workspace expects ($(ws.m), $(ws.n))"
        ),
    )
    length(b) == ws.m || throw(
        DimensionMismatch(
            "b has length $(length(b)), expected $(ws.m)"
        ),
    )
    any(!isfinite, A) && throw(ArgumentError("A contains NaN or Inf entries"))
    any(!isfinite, b) && throw(ArgumentError("b contains NaN or Inf entries"))
    return nothing
end

"""
    validate_nnls_post(A, b, x, w, opts) -> Bool

Post-condition checks for NNLS solution (KKT conditions).
Returns `true` if all checks pass, `false` otherwise.
Logs warnings for each violated condition.
"""
function validate_nnls_post(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    opts::NNLSOptions{T},
) where {T}
    ok = true
    n = length(x)

    # 1. Primal feasibility: x >= 0
    min_x = minimum(x)
    if min_x < -opts.zero_tol
        @warn "NNLS post-condition violation: x contains negative values (min = $min_x)"
        ok = false
    end

    # 2. Dual consistency: w should equal A'(b - Ax)
    r = b - A * x
    w_calc = A' * r
    dual_err = norm(w - w_calc, Inf)
    if dual_err > opts.dual_tol * T(10)
        @warn "NNLS post-condition violation: dual vector w inconsistent (err = $dual_err)"
        ok = false
    end

    # 3. Complementary slackness: x[j] > 0 ⟹ w[j] ≈ 0
    for j in 1:n
        if x[j] > opts.zero_tol && abs(w_calc[j]) > opts.dual_tol * T(100)
            @warn "NNLS post-condition violation: complementarity at j=$j, x=$( x[j]), w=$(w_calc[j])"
            ok = false
            break  # report first violation only
        end
    end

    # 4. Dual feasibility for zero variables: x[j] == 0 ⟹ w[j] <= dual_tol
    for j in 1:n
        if x[j] <= opts.zero_tol && w_calc[j] > opts.dual_tol
            @warn "NNLS post-condition violation: dual feasibility at j=$j, w=$(w_calc[j])"
            ok = false
            break
        end
    end

    return ok
end
