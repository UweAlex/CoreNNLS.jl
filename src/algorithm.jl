# src/algorithm.jl
using LinearAlgebra

"""
    nnls!(ws::NNLSWorkspace, A, b) -> (status, x)

Deterministic Lawson–Hanson NNLS.
Allocations-free inside iteration loop for standard float types.
"""
function nnls!(ws::NNLSWorkspace{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}

    m, n = ws.m, ws.n

    fill!(ws.x, zero(T))
    fill!(ws.passive_set, false)

    ws.r .= b
    mul!(ws.w, transpose(A), ws.r)

    ws.iter = 0

    while true
        ws.iter += 1
        ws.iter > ws.options.max_iter && return (:MaxIter, ws.x)

        # -------------------------------------------------
        # 1. Optimality check
        # -------------------------------------------------
        max_w = zero(T)
        idx_move = 0

        for j in 1:n
            if !ws.passive_set[j]
                wj = ws.w[j]
                if wj > max_w + ws.options.w_tol
                    max_w = wj
                    idx_move = j
                end
            end
        end

        if idx_move == 0
            return (:Success, ws.x)
        end

        ws.passive_set[idx_move] = true

        # -------------------------------------------------
        # 2. Inner loop
        # -------------------------------------------------
        while true

            # count passive
            n_p = 0
            for j in 1:n
                n_p += ws.passive_set[j]
            end

            # build A_p
            col_ptr = 1
            for j in 1:n
                if ws.passive_set[j]
                    @inbounds for i in 1:m
                        ws.A_passive[i, col_ptr] = A[i, j]
                    end
                    col_ptr += 1
                end
            end

            # QR in place
            F = qr!(@view ws.A_passive[:, 1:n_p])

            # -----------------------------
            # Robust rank guard
            # -----------------------------
            # FIX 1: Manually construct R view to avoid getproperty error on Julia 1.6 / SubArrays
            R_view = UpperTriangular(@view F.factors[1:n_p, 1:n_p])
            
            diagmax = zero(T)
            for k in 1:n_p
                diagmax = max(diagmax, abs(R_view[k,k]))
            end

            if diagmax == zero(T)
                ws.passive_set[idx_move] = false
                ws.w[idx_move] = zero(T)
                break
            end

            # Check for rank deficiency
            rank_deficient = false
            for k in 1:n_p
                if abs(R_view[k,k]) <= ws.options.rank_tol * diagmax
                    rank_deficient = true
                    break
                end
            end

            if rank_deficient
                ws.passive_set[idx_move] = false
                ws.w[idx_move] = zero(T)
                break
            end

            # -----------------------------
            # Correct LS solve WITH Generic Fallback
            # -----------------------------

            # copy b into r buffer
            ws.r .= b

            # r <- Q' * b
            # FIX 2: Dispatch based on element type to support BigFloat/Generic
            if T <: Union{Float32, Float64, ComplexF32, ComplexF64}
                # Fast path: LAPACK (in-place, no allocation)
                LinearAlgebra.LAPACK.ormqr!(
                    'L', 'T',
                    F.factors,
                    F.τ,
                    ws.r
                )
            else
                # Generic path: Uses implicit Q multiplication (works for BigFloat)
                # This computes ws.r = Q' * ws.r efficiently without forming Q
                LinearAlgebra.lmul!(LinearAlgebra.adjoint(F.Q), ws.r)
            end

            # solve R * s = r[1:n_p]
            @inbounds for k in 1:n_p
                ws.s[k] = ws.r[k]
            end

            ldiv!(
                R_view,
                @view ws.s[1:n_p]
            )

            # -----------------------------
            # Feasibility check
            # -----------------------------
            feasible = true
            for k in 1:n_p
                if ws.s[k] < -ws.options.w_tol
                    feasible = false
                    break
                end
            end

            if feasible
                col_ptr = 1
                for j in 1:n
                    if ws.passive_set[j]
                        ws.x[j] = ws.s[col_ptr]
                        col_ptr += 1
                    end
                end

                ws.r .= b
                mul!(ws.r, A, ws.x, -one(T), one(T))
                mul!(ws.w, transpose(A), ws.r)

                break
            end

            # -----------------------------
            # Boundary step
            # -----------------------------
            alpha = one(T)
            col_ptr = 1

            for j in 1:n
                if ws.passive_set[j]
                    sj = ws.s[col_ptr]
                    xj = ws.x[j]
                    if sj < -ws.options.w_tol
                        ratio = xj / (xj - sj)
                        alpha = min(alpha, ratio)
                    end
                    col_ptr += 1
                end
            end

            col_ptr = 1
            for j in 1:n
                if ws.passive_set[j]
                    ws.x[j] += alpha * (ws.s[col_ptr] - ws.x[j])
                    col_ptr += 1
                end
            end

            for j in 1:n
                if ws.passive_set[j] && ws.x[j] <= ws.options.w_tol
                    ws.passive_set[j] = false
                    ws.x[j] = zero(T)
                end
            end
        end
    end
end
