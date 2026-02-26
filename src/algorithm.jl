# =============================================================================
# File:    algorithm.jl
# Project: CoreNNLS.jl
# Date:    2025-02
# Purpose: Lawson-Hanson active-set solver for Non-Negative Least Squares (NNLS).
#
#   Solves:   min ‖Ax − b‖₂   subject to   x ≥ 0
#
# Method:  Lawson-Hanson (1974) active-set algorithm with incremental QR.
#
# QR strategy:
#   - On first entry (or after a removal): full column-pivoted QR rebuild — O(m·n_p²)
#   - On each subsequent column addition:  incremental O(m·n_p) column append
#     (avoids rebuilding the existing factorisation from scratch)
#
# References:
#   Lawson, C.L. and Hanson, R.J. (1974). Solving Least Squares Problems.
#   Prentice-Hall. Chapter 23: Algorithm NNLS.
# =============================================================================

"""
    nnls!(ws, A, b) -> NNLSResult

Solve the Non-Negative Least Squares problem

    min ‖Ax − b‖₂   subject to   x ≥ 0

using the Lawson-Hanson active-set algorithm with incremental Householder QR.

The workspace `ws` must have been allocated via `NNLSWorkspace(m, n)` for an
m × n system.  Calling this function is allocation-free after construction.

# Arguments
- `ws` : Pre-allocated `NNLSWorkspace{T}`.
- `A`  : Coefficient matrix (m × n), not modified.
- `b`  : Right-hand side vector (length m), not modified.

# Returns
`NNLSResult` with fields:
- `x`            : Non-negative solution vector (length n).
- `status`       : `:Success` or `:MaxIter`.
- `iterations`   : Number of outer (active-set) iterations performed.
- `residual_norm`: ‖b − Ax‖₂ at the returned solution.
- `kkt_violation`: Maximum KKT complementarity violation max_{j∉P} w[j].

# Example
```julia
ws     = NNLSWorkspace(100, 30)
result = nnls!(ws, A, b)
println(result.x)
```
"""
function nnls!(
    ws::NNLSWorkspace{T}, A::AbstractMatrix{T}, b::AbstractVector{T}
) where {T}

    validate_nnls_inputs(A, b, ws)
    m, n = ws.m, ws.n
    opts = ws.options

    # ------------------------------------------------------------------
    # Tolerance: scale with the largest entry of A and the problem size.
    # Variables with x[j] ≤ tol are considered zero and removed from P.
    # ------------------------------------------------------------------
    max_abs_A = zero(T)
    @inbounds for j in 1:n, i in 1:m
        v = abs(A[i, j])
        if v > max_abs_A; max_abs_A = v; end
    end
    tol = eps(T) * max_abs_A * T(max(m, n))
    if tol == zero(T); tol = eps(T); end

    # ------------------------------------------------------------------
    # Initialise workspace
    # ------------------------------------------------------------------
    fill!(ws.x, zero(T))   # solution estimate x = 0
    reset_passive!(ws)     # passive set P = ∅
    ws.iter = 0

    # qr_n_p: number of columns currently covered by the stored QR.
    # Persists across outer iterations — reset to 0 after any removal.
    qr_n_p = 0

    # Pivoting strategy: Float64/Float32 use column-pivoted QR for better
    # numerical stability and rank detection.  All other types (BigFloat,
    # Rational, …) use unpivoted QR — column norm tracking dominates cost
    # without BLAS acceleration, and higher-precision types need it less.
    use_pivot = T <: Union{Float64, Float32}

    # Initial gradient: w = Aᵀ(b − Ax) = Aᵀb  (since x = 0 initially)
    copyto!(ws.r, b)
    mul!(ws.w, transpose(A), ws.r)

    # ==================================================================
    # Outer loop: move the variable with the largest positive gradient
    # into the passive set, then enforce feasibility via the inner loop.
    # ==================================================================
    while true
        ws.iter += 1

        # --- Max-iteration guard ---
        if ws.iter > opts.max_iter
            copyto!(ws.r, b)
            mul!(ws.r, A, ws.x, -one(T), one(T))   # r = b − Ax
            mul!(ws.w, transpose(A), ws.r)           # w = Aᵀr
            _run_post_checks(ws, A, b)
            return _make_result(ws, :MaxIter)
        end

        # --- Find variable j* with largest positive gradient component ---
        max_w    = zero(T)
        idx_move = 0
        @inbounds for j in 1:n
            if !ws.passive_set[j] && ws.w[j] > max_w
                max_w    = ws.w[j]
                idx_move = j
            end
        end

        # KKT condition satisfied: no active variable has a positive gradient
        if idx_move == 0 || max_w <= zero(T)
            copyto!(ws.r, b)
            mul!(ws.r, A, ws.x, -one(T), one(T))
            mul!(ws.w, transpose(A), ws.r)

            # ----------------------------------------------------------
            # Self-diagnosing quality check via KKT complementarity
            #
            # At a correct solution, the gradient on the passive set must
            # vanish: w[j] = (Aᵀr)[j] ≈ 0 for all j ∈ P.  Any deviation
            # larger than the expected floating-point noise indicates that
            # the triangular solve R⁻¹(Qᵀb) accumulated significant
            # rounding errors — typically caused by ill-conditioned R.
            #
            # Quality tolerance:
            #   quality_tol = eps(T) * max_abs_A * ‖x‖
            # This scales with the problem magnitude and solution norm,
            # giving a dimensionally consistent criterion that requires no
            # hand-tuned thresholds.
            #
            # If the check triggers, we apply iterative refinement using
            # the stored QR — O(m·n_p) cost, not O(m·n).  For accurate
            # solves (Random, Sparse) the check never triggers and costs
            # only a single loop over n_p passive entries.
            # ----------------------------------------------------------
            if qr_n_p > 0
                x_norm = zero(T)
                @inbounds for j in 1:n
                    x_norm += abs2(ws.x[j])
                end
                x_norm = sqrt(x_norm)

                # Quality tolerance: floating-point noise in the triangular
                # solve is O(eps * ‖A‖ * ‖x‖).  We allow a factor-3 margin
                # before declaring the result inaccurate — small enough to
                # catch genuinely bad solves (ill-conditioned R, where
                # passive_grad_max exceeds noise by orders of magnitude),
                # large enough to suppress false positives on well-conditioned
                # problems where passive_grad_max sits just above pure eps noise.
                quality_tol = eps(T) * max_abs_A * x_norm * T(3)

                passive_grad_max = zero(T)
                @inbounds for k in 1:ws.n_passive
                    j = ws.passive_indices[k]
                    v = abs(ws.w[j])
                    if v > passive_grad_max; passive_grad_max = v; end
                end

                if passive_grad_max > quality_tol
                    _refine_qr!(ws, A, b, qr_n_p)
                end
            end

            _run_post_checks(ws, A, b)
            return _make_result(ws, :Success)
        end

        # Move j* into passive set
        add_passive!(ws, idx_move)

        # ==============================================================
        # Inner loop: solve the unconstrained LS on the passive set and
        # remove any variable whose solution is ≤ 0 (restoring feasibility).
        # ==============================================================
        while true
            n_p = ws.n_passive

            # ----------------------------------------------------------
            # Update QR factorisation of the passive submatrix
            # ----------------------------------------------------------
            if n_p == qr_n_p + 1 && qr_n_p > 0
                # APPEND path: one new column has been added since the
                # last QR — extend incrementally in O(m·n_p).
                j_new = ws.passive_indices[n_p]
                householder_append_col!(ws.A_passive, ws.tau,
                                        @view(A[:, j_new]), m, qr_n_p)
                # The appended column gets identity permutation (no pivoting on append)
                ws.perm[n_p] = n_p

                if DEBUG_QR
                    @debug "[APPEND] n_p=$n_p  j_new=$j_new  R[n_p,n_p]=$(ws.A_passive[n_p,n_p])"
                end
            else
                # REBUILD path: triggered on first entry or after a removal.
                # Copy passive columns into A_passive and compute fresh QR.
                @inbounds for col in 1:n_p
                    j = ws.passive_indices[col]
                    for i in 1:m
                        ws.A_passive[i, col] = A[i, j]
                    end
                end
                if use_pivot
                    householder_qr_pivot!(ws.A_passive, ws.tau, ws.perm,
                                          ws.col_norms_sq, m, n_p)
                else
                    # BigFloat etc.: unpivoted QR — cheaper without BLAS norm tracking
                    householder_qr!(ws.A_passive, ws.tau, m, n_p)
                    @inbounds for k in 1:n_p; ws.perm[k] = k; end
                end

                if DEBUG_QR
                    @debug "[REBUILD] n_p=$n_p  previous qr_n_p=$qr_n_p"
                end
            end

            qr_n_p = n_p

            if DEBUG_QR
                A_orig = hcat([A[:, ws.passive_indices[ws.perm[k]]] for k in 1:n_p]...)
                _dbg_check_qr(ws.A_passive, ws.tau, A_orig, m, n_p;
                               label="n_p=$n_p", check_pivot=false)
            end

            # ----------------------------------------------------------
            # Rank check: reject column if R diagonal entry is too small
            # ----------------------------------------------------------
            diagmax = zero(T)
            @inbounds for k in 1:n_p
                d = abs(ws.A_passive[k, k])
                if d > diagmax; diagmax = d; end
            end

            if diagmax == zero(T)
                # Entire passive submatrix is numerically zero — drop column
                remove_passive_at!(ws, n_p)
                ws.w[idx_move] = zero(T)
                qr_n_p = 0
                break
            end

            rank_tol = eps(T) * diagmax
            rank_deficient = false
            @inbounds for k in 1:n_p
                if abs(ws.A_passive[k, k]) <= rank_tol
                    rank_deficient = true; break
                end
            end
            if rank_deficient
                # Column introduces numerical rank deficiency — drop it
                remove_passive_at!(ws, n_p)
                ws.w[idx_move] = zero(T)
                qr_n_p = 0
                break
            end

            # ----------------------------------------------------------
            # Solve unconstrained LS on passive set: s = (AₚᵀAₚ)⁻¹ Aₚᵀ b
            # via the stored QR: s = R⁻¹ (Qᵀb)[1:n_p]
            # Qtb_work is a scratch buffer — keeps ws.r = b − Ax intact.
            # ----------------------------------------------------------
            copyto!(ws.Qtb_work, b)
            householder_apply_qt!(ws.A_passive, ws.tau, ws.Qtb_work, m, n_p)

            @inbounds for i in 1:n_p
                ws.s[i] = ws.Qtb_work[i]
            end

            # ----------------------------------------------------------
            # R-diagonal preconditioning (column equilibration)
            #
            # The condition number of R can be estimated cheaply from its
            # diagonal after column-pivoted QR: since pivoting places the
            # largest-norm columns first, |R[1,1]| ≥ |R[2,2]| ≥ … ≥ |R[n_p,n_p]|.
            # Hence  κ_est = |R[1,1]| / |R[n_p,n_p]|  is a lower bound on κ(R).
            #
            # When κ_est is large (> 1/√eps ≈ 10⁸ for Float64), the standard
            # back-substitution loses accuracy.  We instead solve the equivalent
            # column-equilibrated system:
            #
            #   (R · D) · ŝ = c,   D = diag(1/|R[k,k]|)
            #
            # The diagonal of R·D is ±1, so its condition number is at most
            # κ(R·D) ≤ κ(R) / κ_diag, often several orders of magnitude smaller.
            # After solving we recover s = D · ŝ.
            #
            # This is applied only when needed (κ_est > threshold) to avoid
            # any overhead on well-conditioned problems.
            # ----------------------------------------------------------
            r_diag_min = abs(ws.A_passive[n_p, n_p])
            kappa_est  = (r_diag_min > zero(T)) ? diagmax / r_diag_min : typemax(T)

            # Threshold: activate preconditioning when κ_est > 1/√eps(T)
            # For Float64: threshold ≈ 10⁸ — well below the point where
            # back-substitution starts losing digits, but high enough that
            # well-conditioned problems never pay the preconditioning cost.
            precond_threshold = one(T) / sqrt(eps(T))

            if kappa_est > precond_threshold
                # Compute preconditioning factors d[k] = 1/|R[k,k]|
                @inbounds for k in 1:n_p
                    rk = abs(ws.A_passive[k, k])
                    ws.r_scale[k] = rk > zero(T) ? one(T) / rk : one(T)
                end
                # Preconditioned triangular solve: (R·D)·ŝ = s, then s = D·ŝ
                back_substitute_precond!(ws.A_passive, ws.s, ws.r_scale, n_p)
            else
                # Standard triangular solve — no preconditioning overhead
                back_substitute!(ws.A_passive, ws.s, n_p)   # s = R⁻¹ s
            end

            # ----------------------------------------------------------
            # Feasibility check: if all s[k] > 0, accept as new x
            # ----------------------------------------------------------
            feasible = true
            @inbounds for k in 1:n_p
                if ws.s[k] <= zero(T); feasible = false; break; end
            end

            if feasible
                # Accept solution on passive set; active variables stay at 0
                @inbounds for k in 1:n_p
                    ws.x[ws.passive_indices[ws.perm[k]]] = ws.s[k]
                end
                @inbounds for j in 1:n
                    if !ws.passive_set[j]; ws.x[j] = zero(T); end
                end
                # Update residual and gradient for next outer iteration
                copyto!(ws.r, b)
                mul!(ws.r, A, ws.x, -one(T), one(T))   # r = b − Ax
                mul!(ws.w, transpose(A), ws.r)           # w = Aᵀr
                _dbg_check_solution(ws, A, b; label="feasible")
                break
            end

            # ----------------------------------------------------------
            # Infeasible step: compute the longest feasible step α ∈ (0,1]
            # along the direction (s − x) that keeps x ≥ 0.
            # α = min_{k: s[k]≤0}  x[j] / (x[j] − s[k])
            # ----------------------------------------------------------
            alpha = typemax(T)
            @inbounds for k in 1:n_p
                if ws.s[k] <= zero(T)
                    j   = ws.passive_indices[ws.perm[k]]
                    xj  = ws.x[j]
                    den = xj - ws.s[k]         # > 0 since s[k] ≤ 0 and x[j] ≥ 0
                    if den > zero(T)
                        ratio = xj / den
                        if ratio < alpha; alpha = ratio; end
                    end
                end
            end

            if alpha >= typemax(T) || alpha <= zero(T); break; end

            # Move x towards s by step α
            @inbounds for k in 1:n_p
                j = ws.passive_indices[ws.perm[k]]
                ws.x[j] += alpha * (ws.s[k] - ws.x[j])
            end

            # Remove all passive variables that have been driven to (near) zero
            k = ws.n_passive
            while k >= 1
                j = ws.passive_indices[k]
                if ws.x[j] <= tol
                    ws.x[j] = zero(T)
                    remove_passive_at!(ws, k)
                end
                k -= 1
            end
            qr_n_p = 0   # QR is no longer valid after removals — force rebuild

        end  # inner loop
    end  # outer loop
end



"""
    _refine_qr!(ws, A, b, qr_n_p)

Apply one or two steps of iterative refinement on the passive set using the
already-stored QR factorisation.  Called only when the self-diagnosing KKT
quality check detects that the triangular solve was inaccurate.

Background
----------
After Lawson-Hanson converges, the passive set P is correct but the solution
xₚ = R⁻¹(Qᵀb) may have accumulated rounding errors if R is ill-conditioned
(e.g. Hilbert, Vandermonde, Pascal matrices).  One step of iterative refinement
on the fixed passive set recovers these lost digits at low cost:

    δ = R⁻¹ Qᵀ (b − Aₚxₚ)   — uses stored QR, cost O(m·n_p)
    xₚ ← max(0, xₚ + δ)       — projection keeps x ≥ 0

This is the approach recommended by Lawson & Hanson (1974) §5 for improving
accuracy after active-set convergence.

Cost
----
O(m·n_p) per step via `householder_apply_qt!` + `back_substitute!`.
Compare with O(m·n) for a full gradient step — much cheaper when n_p ≪ n.

# Arguments
- `ws`      : `NNLSWorkspace` with converged solution; ws.r = b − Ax on entry.
- `A`, `b`  : Original problem data.
- `qr_n_p`  : Size of the valid QR currently in ws.A_passive (= ws.n_passive).

# Side effects
Updates ws.x, ws.r, ws.w if refinement improves the residual norm.
Rolls back silently if refinement worsens the solution.
"""
function _refine_qr!(ws::NNLSWorkspace{T}, A::AbstractMatrix{T},
                     b::AbstractVector{T}, qr_n_p::Int) where {T}
    MAX_REFINE = 2
    n_p = qr_n_p
    n_p == 0 && return nothing

    for _ in 1:MAX_REFINE
        r_norm_before = dot(ws.r, ws.r)
        r_norm_before == zero(T) && return nothing   # already exact

        # Step 1: c = Qᵀ r_ls  using scratch buffer (preserves ws.r)
        copyto!(ws.Qtb_work, ws.r)
        householder_apply_qt!(ws.A_passive, ws.tau, ws.Qtb_work, ws.m, n_p)

        # Step 2: δ = R⁻¹ c[1:n_p]
        @inbounds for i in 1:n_p
            ws.s[i] = ws.Qtb_work[i]
        end

        # Use preconditioned solve if R is ill-conditioned
        r_diag_max = abs(ws.A_passive[1,   1])
        r_diag_min = abs(ws.A_passive[n_p, n_p])
        kappa_est  = (r_diag_min > zero(T)) ? r_diag_max / r_diag_min : typemax(T)

        if kappa_est > one(T) / sqrt(eps(T))
            @inbounds for k in 1:n_p
                rk = abs(ws.A_passive[k, k])
                ws.r_scale[k] = rk > zero(T) ? one(T) / rk : one(T)
            end
            back_substitute_precond!(ws.A_passive, ws.s, ws.r_scale, n_p)
        else
            back_substitute!(ws.A_passive, ws.s, n_p)
        end

        # Step 3: xₚ ← max(0, xₚ + δ) and evaluate new residual
        @inbounds for k in 1:n_p
            j = ws.passive_indices[ws.perm[k]]
            ws.x[j] = max(zero(T), ws.x[j] + ws.s[k])
        end

        copyto!(ws.r, b)
        mul!(ws.r, A, ws.x, -one(T), one(T))
        r_norm_after = dot(ws.r, ws.r)

        if r_norm_after >= r_norm_before
            # No improvement — undo correction and stop
            @inbounds for k in 1:n_p
                j = ws.passive_indices[ws.perm[k]]
                ws.x[j] = max(zero(T), ws.x[j] - ws.s[k])
            end
            copyto!(ws.r, b)
            mul!(ws.r, A, ws.x, -one(T), one(T))
            break
        end
        # Accepted — loop for second step if warranted
    end

    # Restore gradient w = Aᵀr for KKT reporting
    mul!(ws.w, transpose(A), ws.r)
    return nothing
end

"""
    _make_result(ws, status) -> NNLSResult

Package the current workspace state into an `NNLSResult`.

# Arguments
- `ws`     : `NNLSWorkspace{T}` with converged or terminated state.
- `status` : `:Success` or `:MaxIter`.

# Returns
`NNLSResult` (allocates a copy of `ws.x`).
"""
function _make_result(ws::NNLSWorkspace{T}, status::Symbol) where {T}
    residual_norm = norm(ws.r)
    # KKT violation: largest positive gradient among active variables
    kkt_violation = zero(T)
    @inbounds for j in 1:ws.n
        if !ws.passive_set[j] && ws.w[j] > kkt_violation
            kkt_violation = ws.w[j]
        end
    end
    return NNLSResult(copy(ws.x), status, ws.iter, residual_norm, kkt_violation)
end