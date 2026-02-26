"""
generate_problems.jl  —  CoreNNLS Benchmark Problem Generator
==============================================================
Generates benchmark/problems.jld2 containing a curated set of NNLS problems
with exact or high-precision reference solutions.

Run once before benchmarking:
    julia --project=. benchmark/generate_problems.jl

Design principles
-----------------
- Constructed problems (class :random, :sparse, :toeplitz):
    x_true = abs.(randn(...))  →  b = A * x_true
    Reference solution is EXACT — no BigFloat needed.

- Ill-conditioned problems (class :hilbert, :vandermonde):
    Reference via BigFloat(256) NNLS.  256-bit gives ~77 decimal digits —
    far beyond Float64 (16 digits) and Float32 (7 digits).  Chosen over
    1024-bit to keep generation time reasonable.

- Float32 variants:
    A subset of problems is stored in Float32 precision to verify that
    CoreNNLS works correctly with lower-precision types.

Problem coverage
----------------
Sizes:   m ∈ [50, 200, 1000, 2000],  n ∈ [10, 50, 200]
         plus larger rectangular cases: 2000×50, 2000×200
Seeds:   3 per (class, m, n) configuration — median over seeds in benchmark
         gives stable, hardware-independent results.
Classes: random, sparse, toeplitz, vandermonde, hilbert
"""

using Pkg; Pkg.add(["JLD2"]; io=devnull)
using LinearAlgebra, Random, JLD2, CoreNNLS

# =============================================================================
# Problem sizes
# =============================================================================

const SIZES = [
    # (m,    n)    — overdetermined (typical NNLS use case)
    (50,    10),
    (50,    50),
    (200,   10),
    (200,   50),
    (200,  200),
    (1000,  10),
    (1000,  50),
    (1000, 200),
    (2000,  50),
    (2000, 200),
]

const SEEDS = [42, 137, 271]   # 3 seeds per configuration

# =============================================================================
# Matrix generators
# =============================================================================

"""
Random Gaussian matrix with constructed solution.
x_true is dense nonneg; exact reference.
"""
function make_random(T, m, n, seed)
    rng  = MersenneTwister(seed)
    A      = T.(randn(rng, m, n))
    x_true = abs.(T.(randn(rng, n))) .+ T(0.1)   # strictly positive, dense
    b      = A * x_true
    return A, b, x_true
end

"""
Sparse random matrix (40% fill) with constructed solution.
x_true is sparse (30% nonzero) — tests active-set selection.
"""
function make_sparse(T, m, n, seed; density_A=0.4, density_x=0.3)
    rng = MersenneTwister(seed)
    A   = zeros(T, m, n)
    for j in 1:n, i in 1:m
        rand(rng) < density_A && (A[i,j] = T(randn(rng)))
    end
    # sparse x_true: fraction density_x of entries nonzero
    x_true = zeros(T, n)
    for j in 1:n
        rand(rng) < density_x && (x_true[j] = abs(T(randn(rng))) + T(0.1))
    end
    # ensure at least one nonzero entry
    if all(iszero, x_true); x_true[1] = one(T); end
    b = A * x_true
    return A, b, x_true
end

"""
Symmetric Toeplitz matrix — common in signal processing and time-series.
First row decays exponentially.  Constructed solution.
"""
function make_toeplitz(T, m, n, seed)
    rng = MersenneTwister(seed)
    # Build m×n Toeplitz: A[i,j] = f(|i-j|)
    decay = T(0.7)
    A = [decay^abs(i-j) for i in 1:m, j in 1:n]
    x_true = abs.(T.(randn(rng, n))) .+ T(0.1)
    b = A * x_true
    return A, b, x_true
end

"""
Vandermonde matrix — highly ill-conditioned for large n.
Reference solution via BigFloat(256) NNLS.
x_true used as construction check but NOT stored as reference
(conditioning may corrupt Float64 solution significantly).
"""
function make_vandermonde(T_store, m, n, seed)
    # Build in BigFloat(256) for accuracy
    setprecision(256) do
        BF = BigFloat
        nodes  = range(BF("0.1"), BF("0.9"), length=m)
        A_bf   = [nodes[i]^(j-1) for i in 1:m, j in 1:n]
        rng    = MersenneTwister(seed)
        x_true = abs.(BF.(randn(rng, n))) .+ BF("0.1")
        b_bf   = A_bf * x_true
        x_ref  = nnls(A_bf, b_bf)   # high-precision NNLS
        return T_store.(A_bf), T_store.(b_bf), T_store.(x_ref)
    end
end

"""
Hilbert matrix — canonical ill-conditioned test case.
Reference via BigFloat(256) NNLS.
"""
function make_hilbert(T_store, k)
    setprecision(256) do
        BF = BigFloat
        A_bf  = [BF(1)/(i+j-1) for i in 1:k, j in 1:k]
        b_bf  = ones(BF, k)
        x_ref = nnls(A_bf, b_bf)
        return T_store.(A_bf), T_store.(b_bf), T_store.(x_ref)
    end
end

# =============================================================================
# Generator dispatch
# =============================================================================

function generate_class(cls, m, n, seed)
    cls == :random      && return make_random(Float64, m, n, seed)
    cls == :sparse      && return make_sparse(Float64, m, n, seed)
    cls == :toeplitz    && return make_toeplitz(Float64, m, n, seed)
    cls == :vandermonde && return make_vandermonde(Float64, m, n, seed)
    cls == :hilbert     && begin
        k = min(m, n, 20)   # Hilbert beyond 20×20 is hopelessly singular
        return make_hilbert(Float64, k)
    end
    error("Unknown class: $cls")
end

problem_name(cls, m, n, seed_idx) =
    cls in (:hilbert,) ? "Hilbert_$(min(m,n,20))x$(min(m,n,20))_s$seed_idx" :
    "$(uppercasefirst(string(cls)))_$(m)x$(n)_s$seed_idx"

# =============================================================================
# Generate all problems
# =============================================================================

println("CoreNNLS Problem Generator")
println("=" ^ 60)
println("Sizes: $(length(SIZES)) configurations × $(length(SEEDS)) seeds")
println("Classes: random, sparse, toeplitz, vandermonde, hilbert")
println()

problems = Dict{String, Any}()
idx = 0

for cls in [:random, :sparse, :toeplitz, :vandermonde, :hilbert]
    global idx
    for (m, n) in SIZES
        # Hilbert is square — only generate for square-ish sizes
        if cls == :hilbert && m != n && min(m,n) > 50
            continue
        end

        for (si, seed) in enumerate(SEEDS)
            name = problem_name(cls, m, n, si)
            haskey(problems, name) && continue
            idx += 1

            print("[$idx] $name ... "); flush(stdout)
            t = @elapsed begin
                A, b, x_ref = generate_class(cls, m, n, seed)
            end

            m_eff, n_eff = size(A)

            # Residual as quality indicator
            res_f64 = norm(b - A * x_ref)

            problems[name] = Dict(
                "A"       => A,
                "b"       => b,
                "x_ref"   => x_ref,
                "cls"     => string(cls),
                "m"       => m_eff,
                "n"       => n_eff,
                "seed"    => seed,
                "ref_res" => res_f64,
                "precision" => "Float64",
            )

            println("$(round(t*1000, digits=1))ms  ‖r‖=$(round(res_f64, sigdigits=3))")
        end
    end
end

# =============================================================================
# Float32 variants — subset for precision testing
# =============================================================================
# Take random and sparse, small/medium sizes — enough to verify correctness.

println()
println("Float32 variants...")
println("-" ^ 40)

F32_SIZES = [(50,10), (200,50), (1000,50)]
F32_SEED  = 42

for cls in [:random, :sparse]
    global idx
    for (m, n) in F32_SIZES
        name = "F32_$(uppercasefirst(string(cls)))_$(m)x$(n)"
        idx += 1
        print("[$idx] $name ... "); flush(stdout)
        t = @elapsed begin
            A32, b32, x_true32 = (cls == :random ?
                make_random(Float32, m, n, F32_SEED) :
                make_sparse(Float32, m, n, F32_SEED))
        end
        res = norm(b32 - A32 * x_true32)
        problems[name] = Dict(
            "A"       => A32,
            "b"       => b32,
            "x_ref"   => x_true32,
            "cls"     => string(cls),
            "m"       => m,
            "n"       => n,
            "seed"    => F32_SEED,
            "ref_res" => Float64(res),
            "precision" => "Float32",
        )
        println("$(round(t*1000, digits=1))ms")
    end
end

# =============================================================================
# Save
# =============================================================================

out_path = joinpath(@__DIR__, "problems.jld2")
jldsave(out_path; problems=problems)

println()
println("=" ^ 60)
println("$(length(problems)) Probleme gespeichert → problems.jld2")
println()
n_f64 = count(v->v["precision"]=="Float64", values(problems))
n_f32 = count(v->v["precision"]=="Float32", values(problems))
println("Float64 Probleme: $n_f64")
println("Float32 Probleme: $n_f32")
println()
println("Klassen:")
for cls in ["random","sparse","toeplitz","vandermonde","hilbert"]
    n_cls = count(v->v["cls"]==cls, values(problems))
    println("  $cls: $n_cls")
end
