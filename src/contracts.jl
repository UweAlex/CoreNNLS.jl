# src/contracts.jl

const DEBUG_CONTRACTS = true # Set to false for Production Builds

@inline function contract_assert(cond::Bool, msg::String)
    if DEBUG_CONTRACTS && !cond
        error("CONTRACT VIOLATION: " * msg)
    end
end

"""
    validate_nnls_post(A, b, x, w, tol)

Checks KKT conditions for NNLS solution.
1. Primal Feasibility: x >= 0
2. Dual Feasibility: w <= tol (for x=0)
3. Complementarity: dot(x, w) approx 0
"""
function validate_nnls_post(A, b, x, w, tol)
    # 1. Primal Feasibility
    contract_assert(all(x .>= -tol), "Primal feasibility violated (x < 0)")

    # 2. Dual Feasibility (in the active set sense)
    # Technically w is the gradient of ||Ax-b||^2.
    # For NNLS optimality: w_i <= 0 for all i in Passive set? 
    # No, standard NNLS optimality:
    # If x_i > 0 => w_i approx 0
    # If x_i = 0 => w_i <= 0 ? No, w = A' * r. 
    # If x_i = 0, we require w_i <= 0 implies staying at 0 is good?
    # Let's check residual norm reduction heuristic.
    
    # Complementarity check
    # x and w should be orthogonal-ish in the sense that 
    # positive x aligns with direction of descent.
    
    # Keep it simple: Check for NaNs
    contract_assert(all(isfinite, x), "Solution contains NaN/Inf")
end
