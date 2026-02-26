# =============================================================================
# File:    householder.jl
# Project: CoreNNLS.jl
# Date:    2025-02
# Purpose: Householder QR factorisation primitives used by the NNLS solver.
#
# Methods:
#   - reflector!            : Compute a Householder reflector in-place.
#   - reflector_apply!      : Apply H = I - τvvᵀ to a matrix (generic + BLAS).
#   - householder_qr!       : Full QR factorisation (unpivoted).
#   - householder_append_col! : Incremental O(m·n) column append — avoids
#                               full O(m·n²) rebuild when one column is added.
#   - householder_apply_qt! : Apply Qᵀ to a vector in-place.
#   - back_substitute!      : Solve upper-triangular system (generic + LAPACK).
#   - householder_qr_pivot! : Column-pivoted QR for rank-revealing factorisation.
#
# All routines are generic over T<:AbstractFloat.
# BLAS/LAPACK specialisations exist for Float64 and Float32.
#
# Debug mode: set DEBUG_QR = true to enable invariant checks (slow).
# =============================================================================

const DEBUG_QR = false

# =============================================================================
# Debug helpers (compiled away when DEBUG_QR = false)
# =============================================================================

"""
    _dbg_check_reflector(x_before, x_after, τ)

Verify that the Householder reflector correctly maps x_before to β·e₁.
Only active when DEBUG_QR = true.

# Arguments
- `x_before` : Vector before applying the reflector.
- `x_after`  : Vector after (contains β in position 1, v[2:end] elsewhere).
- `τ`        : Householder scalar.
"""
function _dbg_check_reflector(x_before::AbstractVector{T},
                               x_after::AbstractVector{T},
                               τ::T) where {T}
    DEBUG_QR || return nothing
    n  = length(x_before)
    β  = x_after[1]
    v  = [one(T); x_after[2:end]]
    H  = Matrix{T}(I, n, n) - τ * v * v'
    res = H * x_before
    err = norm(res - β * [one(T); zeros(T, n-1)])
    if err > 100 * eps(T) * norm(x_before)
        @warn "[DEBUG reflector!] H·x ≠ β·e₁  err=$err  τ=$τ  β=$β  ‖x‖=$(norm(x_before))"
    else
        @debug "[DEBUG reflector!] OK  err=$err"
    end
    orth_err = norm(H'*H - I)
    if orth_err > 100 * eps(T)
        @warn "[DEBUG reflector!] H not orthogonal  err=$orth_err"
    end
    return nothing
end

"""
    _dbg_check_qr(A_passive, tau, A_orig, m, n_p; label="", check_pivot=false)

Verify that the stored QR factorisation satisfies A_orig = Q·[R; 0].
Only active when DEBUG_QR = true.

# Arguments
- `A_passive`   : Compact QR storage (in-place Householder form).
- `tau`         : Householder scalars.
- `A_orig`      : Original matrix (before factorisation).
- `m`, `n_p`    : Problem dimensions.
- `label`       : Optional label for warning messages.
- `check_pivot` : Also verify that |R[k,k]| is non-increasing.
"""
function _dbg_check_qr(A_passive::AbstractMatrix{T},
                        tau::AbstractVector{T},
                        A_orig::AbstractMatrix{T},
                        m::Int, n_p::Int;
                        label::String = "",
                        check_pivot::Bool = false) where {T}
    DEBUG_QR || return nothing
    prefix = isempty(label) ? "[DEBUG QR]" : "[DEBUG QR $label]"
    R = UpperTriangular(A_passive[1:n_p, 1:n_p])
    Q = Matrix{T}(I, m, m)
    for k in 1:n_p
        v = zeros(T, m - k + 1)
        v[1] = one(T)
        v[2:end] .= A_passive[(k+1):m, k]
        Hk = Matrix{T}(I, m-k+1, m-k+1) - tau[k] * v * v'
        Hk_full = Matrix{T}(I, m, m)
        Hk_full[k:m, k:m] = Hk
        Q = Hk_full * Q
    end
    orth_err = norm(Q'*Q - I)
    if orth_err > 1000 * eps(T)
        @warn "$prefix Q not orthogonal  ‖Q'Q-I‖=$orth_err"
    else
        @debug "$prefix Q orthogonal OK  err=$orth_err"
    end
    QR = Q * [Matrix(R); zeros(T, m - n_p, n_p)]
    recon_err = norm(A_orig - QR) / max(norm(A_orig), one(T))
    if recon_err > 1000 * eps(T)
        @warn "$prefix Reconstruction A ≠ QR  rel.err=$recon_err"
    else
        @debug "$prefix Reconstruction OK  rel.err=$recon_err"
    end
    if check_pivot && n_p > 1
        for k in 1:(n_p-1)
            if abs(A_passive[k,k]) < abs(A_passive[k+1,k+1]) - 1000*eps(T)
                @warn "$prefix Pivot rule violated at k=$k"
            end
        end
    end
    return nothing
end

"""
    _dbg_print_passive(ws; label="")

Print the current passive set indices. Only active when DEBUG_QR = true.
"""
function _dbg_print_passive(ws; label::String = "")
    DEBUG_QR || return nothing
    prefix = isempty(label) ? "[DEBUG passive]" : "[DEBUG passive $label]"
    @debug "$prefix n_passive=$(ws.n_passive)  indices=$(ws.passive_indices[1:ws.n_passive])"
    return nothing
end

"""
    _dbg_check_solution(ws, A, b; label="")

Check KKT conditions on the current solution:
  - x ≥ 0
  - w[j] ≈ 0 for passive (j ∈ P)
  - w[j] ≤ 0 for active  (j ∉ P)

Only active when DEBUG_QR = true.

# Arguments
- `ws`    : `NNLSWorkspace` with current x.
- `A`, `b`: Problem data.
- `label` : Optional label for warning messages.
"""
function _dbg_check_solution(ws, A::AbstractMatrix{T},
                              b::AbstractVector{T};
                              label::String = "") where {T}
    DEBUG_QR || return nothing
    prefix = isempty(label) ? "[DEBUG KKT]" : "[DEBUG KKT $label]"
    n = ws.n
    r = b - A * ws.x
    w = A' * r
    neg_x = findall(ws.x .< -1000*eps(T))
    if !isempty(neg_x)
        @warn "$prefix x < 0 at positions $neg_x  min=$(minimum(ws.x[neg_x]))"
    end
    for j in 1:n
        if ws.passive_set[j] && abs(w[j]) > 1000*eps(T)
            @warn "$prefix w[$j]=$(w[j]) should be ≈ 0 (passive variable)"
        end
        if !ws.passive_set[j] && w[j] > 1000*eps(T)
            @warn "$prefix w[$j]=$(w[j]) > 0 (active variable, should be ≤ 0)"
        end
    end
    return nothing
end

# =============================================================================
# Core Householder primitives
# =============================================================================

"""
    reflector!(x) -> τ

Compute a Householder reflector H = I − τvvᵀ that maps `x` to −sign(x[1])‖x‖e₁.

The computation is done in-place:
  - `x[1]` is overwritten with β = −sign(x[1])‖x‖
  - `x[2:end]` are overwritten with the tail of the unit vector v (v[1] = 1 implicitly)

# Arguments
- `x` : Input vector of length ≥ 1 (modified in-place).

# Returns
- `τ` : Householder scalar such that H = I − τvvᵀ is orthogonal.
        Returns 0 if x is already a multiple of e₁.
"""
@inline function reflector!(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    n == 0 && return zero(T)

    @inbounds begin
        ξ₁ = x[1]
        normu_sq = abs2(ξ₁)
        for i in 2:n
            normu_sq += abs2(x[i])
        end

        if normu_sq == zero(T)
            return zero(T)
        end

        normu = sqrt(normu_sq)
        β  = ξ₁ >= zero(T) ? -normu : normu  # choose sign to avoid cancellation
        ξ₁ -= β
        x[1] = β
        inv_ξ₁ = one(T) / ξ₁
        for i in 2:n
            x[i] *= inv_ξ₁                  # store v[2:end] = x[2:end] / (x[1] - β)
        end
    end

    return -ξ₁ / β
end

"""
    reflector_apply!(x, τ, A)

Apply the Householder transformation H = I − τvvᵀ to each column of `A` in-place:
    A ← H · A
where v = [1; x[2:end]] (the first component of v is implicitly 1).

Generic fallback — scalar double loop, works for all AbstractFloat types.

# Arguments
- `x` : Householder vector; v[1] = 1 implicitly, v[2:end] = x[2:end].
- `τ` : Householder scalar.
- `A` : Matrix to update in-place (m × n).

# Returns
`A` (modified in-place).
"""
function reflector_apply!(x::AbstractVector{T}, τ::T,
                          A::AbstractMatrix{T}) where {T<:AbstractFloat}
    m, n = size(A)
    m == 0 && return A
    iszero(τ)  && return A

    @inbounds for j in 1:n
        # dot_val = vᵀ A[:,j]  with v[1] = 1
        dot_val = A[1, j]
        for i in 2:m
            dot_val += x[i] * A[i, j]
        end
        dot_val *= τ
        # A[:,j] -= τ · (vᵀ A[:,j]) · v
        A[1, j] -= dot_val
        for i in 2:m
            A[i, j] -= x[i] * dot_val
        end
    end

    return A
end

"""
    reflector_apply!(x, τ, A)  [Float64/Float32 BLAS specialisation]

Apply H = I − τvvᵀ to `A` using BLAS gemv + ger! for maximum performance.

Algorithm:
  1. w = Aᵀx  via BLAS.gemv (rows 2:m only; row 1 added manually since v[1]=1)
  2. w *= τ
  3. A[1,:] -= w   (first row update)
  4. A[2:m,:] -= x[2:m] · wᵀ  via BLAS.ger! (rank-1 update)

This replaces the scalar double loop with two BLAS calls, giving roughly
2–4× speedup for large matrices.

Size-based dispatch: for small submatrices (m*n < BLAS_THRESHOLD) the fixed
overhead of BLAS call setup exceeds the computation cost.  In that case the
generic scalar fallback is used instead — mathematically identical result,
no accuracy trade-off.

# Arguments
- `x` : Householder vector stored with x[1] = β (not used); v[1] = 1 implicitly.
- `τ` : Householder scalar.
- `A` : Matrix to update in-place (m × n).

# Returns
`A` (modified in-place).
"""
function reflector_apply!(x::AbstractVector{T}, τ::T,
                          A::AbstractMatrix{T}) where {T<:Union{Float64,Float32}}
    m, n = size(A)
    m == 0 && return A
    iszero(τ)  && return A

    # BLAS_THRESHOLD: empirically, BLAS call overhead dominates for m*n < 150.
    # Below this size the scalar generic path is faster (benchmark: m=30 regime).
    # The crossover is hardware-dependent but 150 is conservative for all targets.
    if m * n < 150
        # Scalar fallback — same algorithm as generic reflector_apply!,
        # no BLAS allocation, no call overhead.
        @inbounds for j in 1:n
            dot_val = A[1, j]
            for i in 2:m
                dot_val += x[i] * A[i, j]
            end
            dot_val *= τ
            A[1, j] -= dot_val
            for i in 2:m
                A[i, j] -= x[i] * dot_val
            end
        end
        return A
    end

    # Step 1: w = A[2:m,:]ᵀ · x[2:m]   (n-vector, one BLAS gemv)
    w = BLAS.gemv('T', one(T), @view(A[2:m, :]), @view(x[2:m]))

    # Add contribution from the implicit v[1]=1:  w[j] += A[1,j]
    # Then scale by τ
    @inbounds for j in 1:n
        w[j] += A[1, j]
        w[j] *= τ
    end

    # Step 2: A[1,:] -= wᵀ  (first row; v[1]=1 so scalar update)
    @inbounds for j in 1:n
        A[1, j] -= w[j]
    end

    # Step 3: A[2:m,:] -= x[2:m] · wᵀ  (rank-1 BLAS ger!)
    BLAS.ger!(-one(T), @view(x[2:m]), w, @view(A[2:m, :]))

    return A
end

"""
    householder_qr!(A, tau, m, n_p)

Compute the (unpivoted) QR factorisation of `A[1:m, 1:n_p]` in-place using
successive Householder reflectors.  The result is stored in the standard
compact form: the upper triangle of `A` holds R, and the subdiagonal entries
of each column hold the tail of the corresponding reflector vector.

# Arguments
- `A`    : Matrix (m × ≥n_p); overwritten with compact QR.
- `tau`  : Vector (length ≥ n_p); receives the Householder scalars τₖ.
- `m`    : Number of rows to use.
- `n_p`  : Number of columns to factorise.
"""
function householder_qr!(A::AbstractMatrix{T}, tau::AbstractVector{T},
                          m::Int, n_p::Int) where {T<:AbstractFloat}
    @inbounds for k in 1:min(m, n_p)
        x_before = DEBUG_QR ? copy(A[k:m, k]) : nothing

        τk = reflector!(@view A[k:m, k])
        tau[k] = τk

        if DEBUG_QR
            _dbg_check_reflector(x_before, A[k:m, k], τk)
        end

        if k < n_p
            reflector_apply!(@view(A[k:m, k]), τk, @view(A[k:m, k+1:n_p]))
        end
    end
    return nothing
end

"""
    householder_append_col!(A, tau, col_vec, m, n_p_old)

Incrementally extend an existing QR factorisation by one column at cost O(m·n_p),
avoiding a full O(m·n_p²) rebuild.

After this call, `A[1:m, 1:(n_p_old+1)]` contains the compact QR of the
matrix whose first `n_p_old` columns are represented by the existing factorisation
and whose last column is `col_vec`.

# Arguments
- `A`       : Compact QR storage (m × ≥ n_p_old+1); updated in-place.
- `tau`     : Householder scalars (length ≥ n_p_old+1); tau[n_p_old+1] is set.
- `col_vec` : New column to append (length m).
- `m`       : Number of rows.
- `n_p_old` : Number of columns currently factorised.
"""
function householder_append_col!(A::AbstractMatrix{T}, tau::AbstractVector{T},
                                  col_vec::AbstractVector{T},
                                  m::Int, n_p_old::Int) where {T<:AbstractFloat}
    n_p_new = n_p_old + 1

    # Copy new column into storage position n_p_new
    copyto!(@view(A[1:m, n_p_new]), col_vec)

    # Apply existing reflectors H₁, H₂, …, H_{n_p_old} to the new column
    @inbounds for k in 1:n_p_old
        τk = tau[k]
        iszero(τk) && continue

        # dot_val = vₖᵀ · col[k:m],  vₖ = [1; A[k+1:m, k]]
        tail = m - k
        if tail > 0
            v_tail   = @view A[(k+1):m, k]
            col_tail = @view A[(k+1):m, n_p_new]
            dot_val  = A[k, n_p_new] + dot(v_tail, col_tail)
            dot_val *= τk
            A[k, n_p_new] -= dot_val
            axpy!(-dot_val, v_tail, col_tail)   # BLAS axpy: col_tail -= dot_val * v_tail
        else
            A[k, n_p_new] -= A[k, n_p_new] * τk
        end
    end

    # Compute new reflector for the sub-column starting at position n_p_new
    τ_new = reflector!(@view A[n_p_new:m, n_p_new])
    tau[n_p_new] = τ_new

    if DEBUG_QR
        @debug "[DEBUG append] column $n_p_new appended  τ_new=$τ_new  R[$n_p_new,$n_p_new]=$(A[n_p_new, n_p_new])"
    end

    return nothing
end

"""
    householder_apply_qt!(A, tau, b, m, n_p)

Apply Qᵀ = H_{n_p} · … · H₁ to `b[1:m]` in-place, where Q is stored in the
compact Householder form in `A`.

# Arguments
- `A`    : Compact QR storage; subdiagonal entries hold the reflector tails.
- `tau`  : Householder scalars τₖ (length ≥ n_p).
- `b`    : Right-hand side vector (length m); overwritten with Qᵀb.
- `m`    : Number of rows.
- `n_p`  : Number of reflectors to apply.
"""
function householder_apply_qt!(A::AbstractMatrix{T}, tau::AbstractVector{T},
                                b::AbstractVector{T},
                                m::Int, n_p::Int) where {T<:AbstractFloat}
    @inbounds for k in 1:n_p
        τk = tau[k]
        iszero(τk) && continue

        tail = m - k
        if tail > 0
            v_tail = @view A[(k+1):m, k]
            b_tail = @view b[(k+1):m]
            # dot_val = vₖᵀ · b[k:m]
            dot_val = b[k] + dot(v_tail, b_tail)
            dot_val *= τk
            b[k]   -= dot_val
            axpy!(-dot_val, v_tail, b_tail)
        else
            b[k] -= b[k] * τk
        end
    end
    return nothing
end

"""
    back_substitute!(R, x, n_p)

Solve the upper-triangular system R[1:n_p, 1:n_p] · x[1:n_p] = x[1:n_p]
in-place by back-substitution.

Generic fallback — works for all AbstractFloat types including BigFloat.

# Arguments
- `R`   : Matrix whose upper-left n_p × n_p block is upper triangular.
- `x`   : Right-hand side on entry; solution on exit (length ≥ n_p).
- `n_p` : System size.
"""
function back_substitute!(R::AbstractMatrix{T}, x::AbstractVector{T},
                           n_p::Int) where {T<:AbstractFloat}
    @inbounds for i in n_p:-1:1
        val = x[i]
        for j in (i+1):n_p
            val -= R[i, j] * x[j]
        end
        x[i] = val / R[i, i]
    end
    return nothing
end

"""
    back_substitute!(R, x, n_p)  [Float64/Float32 LAPACK specialisation]

Solve the upper-triangular system via `LAPACK.trtrs!` for large systems, or
via the generic scalar loop for small ones.

LAPACK `trtrs!` uses optimised BLAS routines and is faster for large n_p, but
carries a fixed call overhead that dominates for small passive sets.  The scalar
loop is used when n_p < LAPACK_THRESHOLD, giving the same numerical result with
lower overhead.

LAPACK_THRESHOLD = 12: empirically, trtrs! becomes faster around n_p ≥ 12
(benchmark: small problems with n=8 have n_p ≤ 8, always take scalar path).

# Arguments
Same as the generic version.
"""
function back_substitute!(R::AbstractMatrix{T}, x::AbstractVector{T},
                           n_p::Int) where {T<:Union{Float64,Float32}}
    n_p == 0 && return nothing

    # LAPACK_THRESHOLD: below this, scalar back-substitution is faster due to
    # fixed call overhead in trtrs!.  Accuracy is identical in both paths.
    if n_p < 12
        @inbounds for i in n_p:-1:1
            val = x[i]
            for j in (i+1):n_p
                val -= R[i, j] * x[j]
            end
            x[i] = val / R[i, i]
        end
        return nothing
    end

    # trtrs!('U', 'N', 'N', R, B) solves R·B = B for upper triangular R
    LinearAlgebra.LAPACK.trtrs!('U', 'N', 'N',
        @view(R[1:n_p, 1:n_p]),
        reshape(@view(x[1:n_p]), n_p, 1))
    return nothing
end


"""
    back_substitute_precond!(R, x, d, n_p)

Solve the column-preconditioned upper-triangular system (R·D)·ŝ = x in-place,
then rescale to recover s = D·ŝ, where D = diag(d[1:n_p]).

This is used when the condition number of R is large.  By preconditioning with
D = diag(1/|R[k,k]|) the diagonal of R·D becomes ±1, which dramatically reduces
the condition number of the triangular system and improves the numerical accuracy
of the back-substitution.

Algorithm
---------
For k = n_p, n_p-1, …, 1:
    ŝ[k] = (x[k] − Σ_{j>k} R[k,j]·d[j]·ŝ[j]) / (R[k,k]·d[k])
After the loop:
    x[k] = d[k] · ŝ[k]   (rescale to recover the solution of R·s = x)

# Arguments
- `R`   : Matrix whose upper-left n_p × n_p block is upper triangular (compact QR form).
- `x`   : Right-hand side on entry; solution s of R·s = x on exit (length ≥ n_p).
- `d`   : Preconditioning diagonal; d[k] = 1/|R[k,k]| (length ≥ n_p).
- `n_p` : System size.

# Notes
For well-conditioned R this gives the same result as `back_substitute!`.
For ill-conditioned R the preconditioned path can recover several digits of accuracy.
"""
function back_substitute_precond!(R::AbstractMatrix{T}, x::AbstractVector{T},
                                   d::AbstractVector{T}, n_p::Int) where {T<:AbstractFloat}
    # Preconditioned back-substitution: solve (R·D)·ŝ = x
    # where R̂ = R·D has R̂[i,j] = R[i,j]*d[j], diagonal entries ≈ ±1
    @inbounds for i in n_p:-1:1
        val = x[i]
        for j in (i+1):n_p
            val -= R[i, j] * d[j] * x[j]   # x[j] still holds ŝ[j] (already scaled back)
        end
        # Note: divide by R[i,i]*d[i] = R[i,i]/|R[i,i]| = sign(R[i,i])
        x[i] = val / (R[i, i] * d[i])
    end
    # Rescale: s[k] = d[k] * ŝ[k]
    @inbounds for k in 1:n_p
        x[k] *= d[k]
    end
    return nothing
end

# =============================================================================
# Column-pivoted QR
# =============================================================================

"""
    householder_qr_pivot!(A, tau, perm, col_norms_sq, m, n_p)

Compute a column-pivoted QR factorisation of `A[1:m, 1:n_p]` in-place.
At each step the column with the largest remaining norm is moved to the pivot
position, improving numerical stability for rank-deficient or ill-conditioned
passive submatrices.

After this call:
- `A[1:n_p, 1:n_p]` is upper triangular (R factor, in compact Householder form).
- `tau[1:n_p]` holds the Householder scalars.
- `perm[k]` maps QR position k to the original column index in `passive_indices`.

Column norms are updated incrementally (Golub–Reinsch downdate) to avoid
recomputing them from scratch at each step.

# Arguments
- `A`            : Matrix (m × ≥n_p); overwritten with compact QR.
- `tau`          : Vector (length ≥ n_p); receives Householder scalars.
- `perm`         : Permutation vector (length ≥ n_p); receives pivot mapping.
- `col_norms_sq` : Scratch vector (length ≥ n_p); squared column norms.
- `m`            : Number of rows.
- `n_p`          : Number of columns to factorise.
"""
function householder_qr_pivot!(
    A::AbstractMatrix{T}, tau::AbstractVector{T},
    perm::AbstractVector{Int}, col_norms_sq::AbstractVector{T},
    m::Int, n_p::Int
) where {T<:AbstractFloat}

    # Initialise permutation to identity
    @inbounds for k in 1:n_p
        perm[k] = k
    end

    # Compute initial squared column norms
    @inbounds for j in 1:n_p
        s = zero(T)
        for i in 1:m
            s += abs2(A[i, j])
        end
        col_norms_sq[j] = s
    end

    @inbounds for k in 1:min(m, n_p)

        # --- Pivot: find column with largest remaining norm ---
        if k < n_p
            j_max = k
            for j in (k+1):n_p
                if col_norms_sq[j] > col_norms_sq[j_max]
                    j_max = j
                end
            end

            if j_max != k
                # Swap columns k and j_max in A
                for i in 1:m
                    A[i, k], A[i, j_max] = A[i, j_max], A[i, k]
                end
                col_norms_sq[k], col_norms_sq[j_max] = col_norms_sq[j_max], col_norms_sq[k]
                perm[k], perm[j_max] = perm[j_max], perm[k]
            end
        end

        # --- Householder reflector for column k ---
        τk = reflector!(@view A[k:m, k])
        tau[k] = τk

        if DEBUG_QR
            @debug "[DEBUG pivot QR] k=$k  perm[k]=$(perm[k])  R[k,k]=$(A[k,k])  τ=$τk"
        end

        # --- Apply reflector to remaining columns ---
        if k < n_p
            reflector_apply!(@view(A[k:m, k]), τk, @view(A[k:m, k+1:n_p]))

            # Incremental norm downdate: ‖col[k+1:m]‖² = ‖col‖² − R[k,j]²
            # Rounding errors accumulate over many steps, so every NORM_REFRESH_PERIOD
            # steps we recompute norms exactly from the current subcolumns.
            NORM_REFRESH_PERIOD = 8
            if k % NORM_REFRESH_PERIOD == 0
                # Exact recomputation from rows k+1:m of remaining columns
                for j in (k+1):n_p
                    s = zero(T)
                    for i in (k+1):m
                        s += abs2(A[i, j])
                    end
                    col_norms_sq[j] = s
                end
            else
                for j in (k+1):n_p
                    col_norms_sq[j] -= abs2(A[k, j])
                    if col_norms_sq[j] < zero(T)
                        col_norms_sq[j] = zero(T)  # guard against floating-point underflow
                    end
                end
            end
        end
    end

    if DEBUG_QR
        @debug "[DEBUG pivot QR] perm[1:$n_p]=$(perm[1:n_p])"
        diag = [A[k,k] for k in 1:min(m,n_p)]
        @debug "[DEBUG pivot QR] |R diagonal|=$(abs.(diag))"
    end

    return nothing
end