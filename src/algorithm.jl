"""
    nnls!(ws::NNLSWorkspace, A, b) -> (status::Symbol, x::Vector)

Deterministic Lawson–Hanson NNLS solver (in-place).

Solves  min ||Ax - b||₂  subject to  x ≥ 0.

Mutates `ws` and returns `(status, ws.x)` where `status` is one of:
- `:Success`  — KKT conditions satisfied within tolerance
- `:MaxIter`  — iteration limit reached (solution may be suboptimal)

# Pre-conditions
- `size(A) == (ws.m, ws.n)` and `length(b) == ws.m`
- All entries of `A` and `b` must be finite
"""
function nnls!(
    ws::NNLSWorkspace{T}, A::AbstractMatrix{T}, b::AbstractVector{T}
) where {T}

    # ------------------------------------------------------------------
    # 0. Input validation
    # ------------------------------------------------------------------
    validate_nnls_inputs(A, b, ws)

    m, n = ws.m, ws.n
    opts = ws.options

    # ------------------------------------------------------------------
    # 1. Initialisation
    # ------------------------------------------------------------------
    fill!(ws.x, zero(T))
    reset_passive!(ws)

    # r = b - A*x = b  (since x = 0)
    ws.r .= b
    # w = A' * r  (dual / gradient)
    mul!(ws.w, transpose(A), ws.r)

    ws.iter = 0

    # ------------------------------------------------------------------
    # 2. Main (outer) loop — add one variable at a time to passive set
    # ------------------------------------------------------------------
    while true
        ws.iter += 1
        if ws.iter > opts.max_iter
            _run_post_checks(ws, A, b)
            return (:MaxIter, ws.x)
        end

        # ----- Optimality check: find max dual in active set -----
        max_w   = zero(T)
        idx_move = 0
        for j in 1:n
            if !ws.passive_set[j]
                wj = ws.w[j]
                if wj > max_w + opts.dual_tol
                    max_w  = wj
                    idx_move = j
                end
            end
        end

        # All duals non-positive in active set → optimal
        if idx_move == 0
            _run_post_checks(ws, A, b)
            return (:Success, ws.x)
        end

        # Move variable idx_move into passive set
        add_passive!(ws, idx_move)

        # ----- Inner loop: solve LS on passive set, enforce feasibility -----
        while true
            n_p = ws.n_passive

            # Build A_passive from passive index list (O(m * n_p))
            for (col, j) in enumerate(@view ws.passive_indices[1:n_p])
                @inbounds for i in 1:m
                    ws.A_passive[i, col] = A[i, j]
                end
            end

            # QR factorisation of passive sub-matrix
            # NOTE: qr! on a view allocates the QRCompactWY struct internally.
            #       For truly zero-alloc inner loops a hand-rolled Householder
            #       with pre-allocated tau would be needed.
            F = qr!(@view ws.A_passive[:, 1:n_p])

            # ---- Rank guard ----
            R_view = UpperTriangular(@view F.factors[1:n_p, 1:n_p])
            diagmax = zero(T)
            for k in 1:n_p
                diagmax = max(diagmax, abs(R_view[k, k]))
            end

            if diagmax == zero(T)
                # Completely singular — back out the last variable
                remove_passive!(ws, idx_move)
                ws.w[idx_move] = zero(T)   # prevent cycling
                break
            end

            rank_deficient = false
            for k in 1:n_p
                if abs(R_view[k, k]) <= opts.rank_tol * diagmax
                    rank_deficient = true
                    break
                end
            end
            if rank_deficient
                remove_passive!(ws, idx_move)
                ws.w[idx_move] = zero(T)   # prevent cycling
                break
            end

            # ---- LS solve: s = R⁻¹ Q' b ----
            ws.r .= b
            # Apply Q' to r  (dispatch for performance)
            if T <: Union{Float32, Float64, ComplexF32, ComplexF64}
                LinearAlgebra.lmul!(LinearAlgebra.adjoint(F.Q), ws.r)
            else
                # Generic path (BigFloat etc., may allocate)
                ws.r .= LinearAlgebra.adjoint(F.Q) * ws.r
            end

            @inbounds for k in 1:n_p
                ws.s[k] = ws.r[k]
            end
            ldiv!(R_view, @view ws.s[1:n_p])

            # ---- Feasibility check ----
            feasible = true
            for k in 1:n_p
                if ws.s[k] < -opts.feas_tol
                    feasible = false
                    break
                end
            end

            if feasible
                # Accept solution — scatter s → x via passive index list
                for (k, j) in enumerate(@view ws.passive_indices[1:n_p])
                    ws.x[j] = ws.s[k]
                end
                # Update residual and dual
                ws.r .= b
                mul!(ws.r, A, ws.x, -one(T), one(T))   # r = b - A*x
                mul!(ws.w, transpose(A), ws.r)            # w = A' * r
                break
            end

            # ---- Boundary step (interpolation towards feasibility) ----
            alpha = one(T)
            for (k, j) in enumerate(@view ws.passive_indices[1:n_p])
                sj = ws.s[k]
                xj = ws.x[j]
                if sj < -opts.feas_tol
                    ratio = xj / (xj - sj)
                    alpha = min(alpha, ratio)
                end
            end

            # Interpolate:  x ← x + α(s - x)
            for (k, j) in enumerate(@view ws.passive_indices[1:n_p])
                ws.x[j] += alpha * (ws.s[k] - ws.x[j])
            end

            # Clamp near-zero passives back to active set
            # Iterate in reverse because remove_passive! compacts the list
            for idx in ws.n_passive:-1:1
                j = ws.passive_indices[idx]
                if ws.x[j] <= opts.zero_tol
                    remove_passive!(ws, j)
                    ws.x[j] = zero(T)
                end
            end
        end # inner loop
    end # outer loop
end

# --------------------------------------------------------------------------
# Optional post-condition check
# --------------------------------------------------------------------------
function _run_post_checks(ws::NNLSWorkspace{T}, A, b) where {T}
    if ws.options.check_contracts
        validate_nnls_post(A, b, ws.x, ws.w, ws.options)
    end
end
