using LinearAlgebra

"""
    nnls!(ws::NNLSWorkspace, A, b) -> (status, x)

Solves min ||Ax - b|| s.t. x >= 0 using the Lawson-Hanson algorithm.
Deterministic, in-place, standard QR.
"""
function nnls!(ws::NNLSWorkspace{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    
    # --- 0. Initialization & Reset ---
    m, n = ws.m, ws.n
    fill!(ws.x, zero(T))
    fill!(ws.passive_set, false) # All start in Active Set Z
    
    # Initial Residual: r = b (since Ax=0)
    ws.r .= b
    
    # Initial Dual: w = A'b
    mul!(ws.w, transpose(A), ws.r)
    
    ws.iter = 0
    
    # --- Main Loop ---
    while true
        ws.iter += 1
        if ws.iter > ws.options.max_iter
            @warn "NNLS max iterations reached"
            return (:MaxIter, ws.x)
        end

        # --- 1. Optimality Check ---
        # Find index j in Z with maximum w_j.
        # We want to move x_i up if w_i > 0.
        max_w = zero(T)
        idx_move = 0
        
        # Deterministic scan: first index wins in case of tie
        for j in 1:n
            if !ws.passive_set[j]
                wj = ws.w[j]
                if wj > max_w
                    max_w = wj
                    idx_move = j
                end
            end
        end
        
        # Termination Condition
        if max_w <= ws.options.w_tol || idx_move == 0
            # Optimal found
            # Post-check (debug) - only if contracts are enabled/debug mode
            # validate_nnls_post(A, b, ws.x, ws.w, ws.options.w_tol)
            return (:Success, ws.x)
        end

        # --- 2. Move to Passive Set ---
        ws.passive_set[idx_move] = true
        
        # --- 3. Inner Loop: Solve Least Squares for Passive Set ---
        while true
            n_passive = count(ws.passive_set)
            
            # Build Submatrix A_passive = A[:, P]
            # Copy into pre-allocated buffer
            col_idx = 1
            for j in 1:n
                if ws.passive_set[j]
                    for i in 1:m
                        ws.A_passive[i, col_idx] = A[i, j]
                    end
                    col_idx += 1
                end
            end
            
            # QR Decomposition of A_passive
            F = qr!(@view ws.A_passive[:, 1:n_passive])
            
            # --- 3a. Rank Guard (CORRECTED) ---
            R_diag = diag(F.R)
            if any(abs.(R_diag) .< ws.options.rank_tol)
                # Rank deficiency detected!
                # Strategy: We cannot add this column. 
                # FIX: Set w[idx_move] = 0 to prevent immediate re-selection
                # and move it back to Z.
                ws.w[idx_move] = zero(T) 
                ws.passive_set[idx_move] = false 
                break # Break inner loop, continue outer loop
            end
            
            # Solve LS: min || A_p * s - b ||
            fill!(ws.s, zero(T))
            s_view = @view ws.s[1:n_passive]
            
            # Solve R * s = Q' * b
            ldiv!(s_view, F, b)
            
            # --- 4. Feasibility Check ---
            if all(s_view .>= zero(T))
                # Feasible step found! Update x.
                col_idx = 1
                for j in 1:n
                    if ws.passive_set[j]
                        ws.x[j] = ws.s[col_idx]
                        col_idx += 1
                    end
                end
                
                # Update Residual r = b - A*x
                ws.r .= b
                mul!(ws.r, A, ws.x, -one(T), one(T))
                
                # Update Dual w = A'r
                mul!(ws.w, transpose(A), ws.r)
                
                break # Exit Inner Loop
            end
            
            # --- 5. Boundary Step ---
            alpha = one(T)
            
            # Map passive indices to vector s
            # Note: findall allocates. For Phase 1 acceptable, 
            # for Phase 3 SLSQP consider maintaining an index list in workspace.
            passive_indices = findall(ws.passive_set)
            
            # Find minimum ratio alpha
            for (k, j) in enumerate(passive_indices)
                sj = ws.s[k]
                xj = ws.x[j]
                
                # Only consider variables that would go negative (sj < 0)
                if sj < zero(T)
                    # Ratio xj / (xj - sj)
                    # Example: x=10, s=-10 -> ratio = 10/20 = 0.5 -> x_new = 0
                    ratio = xj / (xj - sj)
                    if ratio < alpha
                        alpha = ratio
                    end
                end
            end
            
            # Apply Step: x = x + alpha * (s - x)
            for (k, j) in enumerate(passive_indices)
                ws.x[j] += alpha * (ws.s[k] - ws.x[j])
            end
            
            # Move zeroed out variables back to Z
            # Use a small tolerance to detect 'zero'
            for j in passive_indices
                if ws.x[j] < ws.options.w_tol 
                    ws.passive_set[j] = false
                    ws.x[j] = zero(T) 
                end
            end
            
            # Inner loop continues with reduced Passive Set
        end
    end
end
