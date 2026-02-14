# src/algorithm.jl
using LinearAlgebra

"""
    nnls!(ws::NNLSWorkspace, A, b) -> (status, x)

Solves min ||Ax - b|| s.t. x >= 0 using a deterministic Lawson-Hanson implementation.
"""
function nnls!(ws::NNLSWorkspace{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    
    # --- 0. Initialization ---
    m, n = ws.m, ws.n
    fill!(ws.x, zero(T))
    fill!(ws.passive_set, false)
    ws.r .= b
    mul!(ws.w, transpose(A), ws.r)
    ws.iter = 0
    
    while true
        ws.iter += 1
        (ws.iter > ws.options.max_iter) && return (:MaxIter, ws.x)

        # --- 1. Optimality Check (Deterministic Scan) ---
        max_w = -typemax(T) # Start at negative infinity for correct max search
        idx_move = 0
        
        for j in 1:n
            if !ws.passive_set[j]
                wj = ws.w[j]
                if wj > max_w
                    max_w = wj
                    idx_move = j
                end
            end
        end
        
        # Termination: No positive gradient component left
        if idx_move == 0 || max_w <= ws.options.w_tol
            return (:Success, ws.x)
        end

        # --- 2. Move to Passive Set ---
        ws.passive_set[idx_move] = true
        
        # --- 3. Inner Loop: Solve Least Squares ---
        while true
            n_p = count(ws.passive_set)
            
            # Build Submatrix A_p (In-place copy)
            col_ptr = 1
            for j in 1:n
                if ws.passive_set[j]
                    for i in 1:m
                        ws.A_passive[i, col_ptr] = A[i, j]
                    end
                    col_ptr += 1
                end
            end
            
            # QR Decomposition (In-place on buffer)
            F = qr!(@view ws.A_passive[1:m, 1:n_p])
            
            # --- 3a. Rank-Deficiency Guard (Allokationsfrei) ---
            # Check diagonal of R relative to the first element
            r11 = abs(F.R[1,1])
            is_singular = false
            for k in 1:n_p
                if abs(F.R[k,k]) <= ws.options.rank_tol * r11
                    is_singular = true; break
                end
            end

            if is_singular
                # Backtrack: move idx_move back to Z, zero its dual to prevent cycling
                ws.passive_set[idx_move] = false
                ws.w[idx_move] = zero(T) 
                break # Exit inner loop, pick next best w in outer loop
            end
            
            # Solve R * s = Q' * b
            s_view = @view ws.s[1:n_p]
            ldiv!(s_view, F, b)
            
            # --- 4. Feasibility Check (With Tolerance) ---
            # We check if s_view >= -tol to avoid noise-driven boundary steps
            is_feasible = true
            for k in 1:n_p
                if s_view[k] < -ws.options.w_tol
                    is_feasible = false; break
                end
            end

            if is_feasible
                # Accept step: Update x from s
                col_ptr = 1
                for j in 1:n
                    if ws.passive_set[j]
                        ws.x[j] = ws.s[col_ptr]
                        col_ptr += 1
                    end
                end
                # UPDATE RESIDUAL AND DUAL (Crucial for next outer iteration)
                ws.r .= b
                mul!(ws.r, A, ws.x, -one(T), one(T))
                mul!(ws.w, transpose(A), ws.r)
                break # Success in inner loop
            end
            
            # --- 5. Boundary Step (Variables hitting zero) ---
            alpha = one(T)
            col_ptr = 1
            idx_boundary = 0
            
            for j in 1:n
                if ws.passive_set[j]
                    sj = ws.s[col_ptr]
                    xj = ws.x[j]
                    if sj < -ws.options.w_tol # Component wants to become negative
                        ratio = xj / (xj - sj)
                        if ratio < alpha
                            alpha = ratio
                            idx_boundary = j
                        end
                    end
                    col_ptr += 1
                end
            end
            
            # Interpolate: x = x + alpha * (s - x)
            col_ptr = 1
            for j in 1:n
                if ws.passive_set[j]
                    ws.x[j] += alpha * (ws.s[col_ptr] - ws.x[j])
                    col_ptr += 1
                end
            end
            
            # Move zeroed variables back to Active Set Z
            # Forensic Fix: Ensure values < tol are strictly zeroed
            for j in 1:n
                if ws.passive_set[j] && ws.x[j] < ws.options.w_tol
                    ws.passive_set[j] = false
                    ws.x[j] = zero(T)
                end
            end
            
            # IMPORTANT: In LH-Algorithm, we stay in inner loop after boundary step
            # but we need to re-build A_p for the reduced passive set.
        end
    end
end
