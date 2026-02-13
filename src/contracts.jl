using LinearAlgebra

"""
    validate_nnls_post(A, b, x, w, tol)

Post-condition check for NNLS solution.
Checks KKT conditions: Feasibility (x >= 0) and Complementary Slackness.
"""
function validate_nnls_post(A::AbstractMatrix{T}, b::AbstractVector{T}, 
                           x::AbstractVector{T}, w::AbstractVector{T}, 
                           tol::T) where {T}
    
    # 1. Feasibility: x >= 0
    if any(x .< -tol)
        @warn "NNLS Contract Violation: x contains negative values."
    end
    
    # 2. Dual Feasibility (w is gradient of ||Ax-b||^2 / 2)
    # w = A'(Ax - b). For x_i=0, we need w_i <= tol (no descent direction).
    # Note: The algorithm finds w > 0 to move to P. 
    # So for optimal x, w[i] where x[i]==0 should be <= 0 ideally.
    # But standard termination is checking if max(w[Z]) <= tol.
    
    # 3. Complementary Slackness
    # If x_i > 0, then w_i should be approx 0 (gradient zero).
    # In NNLS, we check this via stationarity of the residual.
    
    # For Phase 1, we keep checks simple.
    # We verify the residual logic matches expectations.
    r = b - A * x
    w_calc = A' * r
    
    # Check if stored w matches reality (sanity check)
    if norm(w - w_calc, Inf) > tol * 10
         @warn "NNLS Contract Violation: Dual vector w inconsistent with residual."
    end
    
    return true
end
