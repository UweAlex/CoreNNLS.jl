# src/algorithm.jl
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
        # Find index j in Z with maximum w_j (gradient)
        # We look for the most violated constraint (most negative gradient? No.)
        # Standard Lawson-Hanson:
        # w = A'(b - Ax).
        # We want to move x_i up if w_i > 0 (gradient allows descent).
        # So we look for max(w_j) where j in Z.
        
        max_w = zero(T)
        idx_move = 0
        
        # Deterministic scan: argmax with tie-breaking (first index wins)
        # Note: In Julia, argmax finds the first index automatically.
        # But we need to filter by Active Set.
        
        # Efficient scan
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
            # Post-check (debug)
            validate_nnls_post(A, b, ws.x, ws.w, ws.options.w_tol)
            return (:Success, ws.x)
        end

        # --- 2. Move to Passive Set ---
        ws.passive_set[idx_move] = true
        
        # --- 3. Inner Loop: Solve Least Squares for Passive Set ---
        while true
            # Count passive elements
            n_passive = count(ws.passive_set)
            
            # Build Submatrix A_passive = A[:, P]
            # We copy into pre-allocated buffer
            # (This copy is acceptable; it's faster than reallocating the matrix struct)
            col_idx = 1
            for j in 1:n
                if ws.passive_set[j]
                    for i in 1:m
                        ws.A_passive[i, col_idx] = A[i, j]
                    end
                    col_idx += 1
                end
            end
            
            # QR Decomposition of A_passive (Standard, no pivoting)
            # We use in-place QR on the buffer
            F = qr!(@view ws.A_passive[:, 1:n_passive])
            
            # --- 3a. Rank Guard ---
            # Check diagonal of R. If zero/near-zero, matrix is rank deficient.
            R_diag = diag(F.R)
            if any(abs.(R_diag) .< ws.options.rank_tol)
                # Rank deficiency detected!
                # Strategy: Remove the last added column from Passive Set P
                # and put it back into Active Set Z.
                # This matches forensic "Give up on this direction" behavior.
                ws.passive_set[idx_move] = false 
                break # Break inner loop, continue outer loop (will pick next best w)
            end
            
            # Solve LS: min || A_p * s - b || 
            # s is the solution for the Passive variables.
            # b is in ws.r currently? No, ws.r is residual. Target is original b.
            # Let's compute target z = b for now. 
            # Actually standard approach: Solve for s using current residual?
            # No, solve full system A_p * s = b.
            
            # Solve R * s = Q' * b
            # ldiv! handles this efficiently
            fill!(ws.s, zero(T)) # Clear full s
            s_view = @view ws.s[1:n_passive]
            
            # Compute Q'b into s_view
            ldiv!(s_view, F, b)
            
            # --- 4. Feasibility Check ---
            # Check if s < 0
            if all(s_view .>= zero(T))
                # Feasible step found! Update x.
                # Map s back to x
                col_idx = 1
                for j in 1:n
                    if ws.passive_set[j]
                        ws.x[j] = ws.s[col_idx]
                        col_idx += 1
                    end
                end
                
                # Update Residual r = b - A*x
                # (Optimization: r = b - A_p * s, but full mul is safer for drift)
                ws.r .= b
                mul!(ws.r, A, ws.x, -one(T), one(T))
                
                # Update Dual w = A'r
                mul!(ws.w, transpose(A), ws.r)
                
                break # Exit Inner Loop -> Continue Outer Loop
            end
            
            # --- 5. Boundary Step ---
            # Some components of s are negative. We must shorten the step.
            # Find ratio alpha = x_j / (x_j - s_j)
            alpha = one(T)
            
            for j in 1:n
                if ws.passive_set[j]
                    # Note: s_view index mapping is needed
                    # But ws.s has values at the front. 
                    # We need to map back to find indices.
                end
            end

            # Simpler approach for Boundary Step using full vectors for clarity:
            # We iterate over passive indices
            # alpha = min( x_j / (x_j - s_j) ) for x_j != s_j and s_j < 0
            
            for j in 1:n
                if ws.passive_set[j]
                    # Find corresponding s value (needs better tracking)
                    # Optimization: Simple search for now to keep code readable
                    # Find s_j inside ws.s
                    # ... This mapping is complex in a flat array ...
                end
            end
            
            # Let's implement the logic more robustly:
            # We need the indices of passive variables to map s to them.
            passive_indices = findall(ws.passive_set)
            
            for (k, j) in enumerate(passive_indices)
                sj = ws.s[k] # The k-th element in s corresponds to variable j
                xj = ws.x[j]
                
                if sj < zero(T)
                    ratio = xj / (xj - sj)
                    if ratio < alpha
                        alpha = ratio
                    end
                end
            end
            
            # Apply Step: x = x + alpha * (s - x)
            # Only update passive indices
            for (k, j) in enumerate(passive_indices)
                ws.x[j] += alpha * (ws.s[k] - ws.x[j])
            end
            
            # Move zeroed out variables back to Z
            for j in passive_indices
                if ws.x[j] < ws.options.w_tol # Treat near-zero as zero
                    ws.passive_set[j] = false
                    ws.x[j] = zero(T) # Explicitly set to 0
                end
            end
            
            # Loop Inner continues (re-solve with new Passive Set)
        end
    end
end
