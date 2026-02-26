"""
Benchmark.jl  —  CoreNNLS Final Quality Benchmark
===================================================
Loads all problems from benchmark/problems.jld2 (generated once by
generate_problems.jl) and reports:

  1. Main table:  CoreNNLS vs NonNegLeastSquares (Float64)
  2. Scaling:     Median speedup by m and by n
  3. Precision:   CoreNNLS with Float32 and BigFloat — correctness & speed

Metrics
-------
  Speedup     = t_NonNeg / t_Core        (>1 = CoreNNLS faster)
  Accuracy    = ‖Δx‖_NonNeg / ‖Δx‖_Core  (>1 = CoreNNLS more accurate)
  ‖r‖         = ‖b − Ax‖ / ‖b‖           (relative residual, precision section)
"""

using LinearAlgebra, Statistics, BenchmarkTools
using NonNegLeastSquares, CoreNNLS, JLD2

prob_path = joinpath(@__DIR__, "problems.jld2")
if !isfile(prob_path)
    println("⚠  problems.jld2 nicht gefunden.")
    println("   Bitte zuerst ausführen:")
    println("   julia --project=. benchmark/generate_problems.jl")
    exit(1)
end

all_problems = load(prob_path, "problems")

# Separate Float64 and Float32 problems
f64_problems = filter(p -> p.second["precision"] == "Float64", all_problems)
f32_problems = filter(p -> p.second["precision"] == "Float32", all_problems)

# =============================================================================
# 1. Main comparison table — Float64 only, exclude ill-conditioned structured
#    matrices from speedup median (they have tiny passive sets)
# =============================================================================

println("# CoreNNLS — Finaler Qualitätsbenchmark")
println()
println("Speedup     = t_NonNeg / t_Core        (>1 = CoreNNLS schneller)")
println("Genauigkeit = ‖Δx‖_NonNeg / ‖Δx‖_Core  (>1 = CoreNNLS genauer)")
println()

println("## Ergebnisse (Float64)")
println()
println("| Problem | m | n | Speedup | Genauigkeit |")
println("|:--------|--:|--:|--------:|------------:|")

all_speedups   = Float64[]
all_accuracy   = Float64[]
speedup_by_m   = Dict{Int,Vector{Float64}}()
speedup_by_n   = Dict{Int,Vector{Float64}}()

for (name, prob) in sort(collect(f64_problems), by=x->x[1])
    A     = prob["A"]
    b     = prob["b"]
    x_ref = prob["x_ref"]
    m, n  = prob["m"], prob["n"]
    cls   = prob["cls"]

    # Warm up
    ws = NNLSWorkspace(m, n)
    nnls!(ws, copy(A), copy(b))
    nonneg_lsq(copy(A), copy(b); alg=:nnls)

    tc = @belapsed nnls!($ws, copy($A), copy($b))
    tn = @belapsed nonneg_lsq(copy($A), copy($b); alg=:nnls)

    x_c = nnls(copy(A), copy(b))
    x_n = vec(nonneg_lsq(copy(A), copy(b); alg=:nnls))

    err_c = norm(x_c - x_ref)
    err_n = norm(x_n - x_ref)

    speedup  = tn / tc
    accuracy = (err_c > 1e-300 && err_n > 1e-300) ? err_n / err_c :
               (err_c < 1e-300 && err_n < 1e-300) ? 1.0 :
               (err_c < 1e-300) ? Inf : 0.0

    push!(all_speedups, speedup)
    isfinite(accuracy) && push!(all_accuracy, accuracy)

    if !(cls in ("hilbert", "vandermonde"))
        haskey(speedup_by_m, m) || (speedup_by_m[m] = Float64[])
        haskey(speedup_by_n, n) || (speedup_by_n[n] = Float64[])
        push!(speedup_by_m[m], speedup)
        push!(speedup_by_n[n], speedup)
    end

    sp_str = string(round(speedup,  digits=2))
    gn_str = isinf(accuracy) ? "∞" :
             isnan(accuracy) ? "—" :
             string(round(accuracy, digits=2))

    println("| $name | $m | $n | $sp_str | $gn_str |")
end

# =============================================================================
# 2. Summary
# =============================================================================

println()
println("## Zusammenfassung (Float64)")
println()
println("| Metrik | Wert |")
println("|:-------|-----:|")
println("| Median Speedup          | $(round(median(all_speedups), digits=2)) |")
println("| Median Genauigkeit      | $(round(median(all_accuracy), digits=2)) |")
println("| CoreNNLS schneller (>1) | $(count(all_speedups .> 1))/$(length(all_speedups)) |")
println("| CoreNNLS genauer   (>1) | $(count(all_accuracy .> 1))/$(length(all_accuracy)) |")

# =============================================================================
# 3. Scaling analysis
# =============================================================================

println()
println("## Skalierung — Median Speedup nach Dimension")
println("(nur random/sparse/toeplitz, über alle Seeds gemittelt)")
println()

println("### Nach m")
println()
println("| m | Median Speedup | Anzahl |")
println("|--:|---------------:|-------:|")
for m in sort(collect(keys(speedup_by_m)))
    v = speedup_by_m[m]
    println("| $m | $(round(median(v), digits=2)) | $(length(v)) |")
end

println()
println("### Nach n")
println()
println("| n | Median Speedup | Anzahl |")
println("|--:|---------------:|-------:|")
for n in sort(collect(keys(speedup_by_n)))
    v = speedup_by_n[n]
    println("| $n | $(round(median(v), digits=2)) | $(length(v)) |")
end

# =============================================================================
# 4. Precision benchmark — Float32 and BigFloat
#    Verifies CoreNNLS works correctly with non-Float64 types.
# =============================================================================

println()
println("## Präzisionstest")
println()
println("Zeigt dass CoreNNLS korrekt mit Float32 und BigFloat(128) arbeitet.")
println("‖r‖_rel = ‖b − Ax‖ / ‖b‖  (relative Residualnorm)")
println()

# --- Float32 ---
println("### Float32")
println()
println("| Problem | m | n | ‖r‖_rel | Status |")
println("|:--------|--:|--:|--------:|-------:|")

for (name, prob) in sort(collect(f32_problems), by=x->x[1])
    A32   = prob["A"]
    b32   = prob["b"]
    m, n  = prob["m"], prob["n"]

    result = nnls(copy(A32), copy(b32))
    rrel   = norm(b32 - A32 * result) / norm(b32)
    status = rrel < 1f-3 ? "✓" : "⚠"

    println("| $name | $m | $n | $(round(Float64(rrel), sigdigits=3)) | $status |")
end

# --- BigFloat(128) — moderate precision, fast ---
println()
println("### BigFloat (128-bit, ~38 Dezimalstellen)")
println()
println("Alle drei Solver in BigFloat(128) — kein BLAS, reiner Algorithmusvergleich.")
println("Speedup  = t_NonNeg / t_Core  (>1 = CoreNNLS schneller)")
println("Genauigkeit = ‖r‖_NonNeg / ‖r‖_Core  (>1 = CoreNNLS genauer)")
println()
println("| Problem | m | n | Speedup BF | Genauigkeit BF | ‖r‖ Core BF | ‖r‖ Core F64 |")
println("|:--------|--:|--:|-----------:|---------------:|------------:|-------------:|")

# Representative subset: random + vandermonde, one seed, small/medium sizes
# (BigFloat is slow — keep subset manageable)
bf_subset = filter(p -> p.second["cls"] in ("random","vandermonde") &&
                        p.second["precision"] == "Float64" &&
                        p.second["m"] <= 200 &&
                        p.second["n"] <= 50 &&
                        endswith(p.first, "_s1"),
                   f64_problems)

for (name, prob) in sort(collect(bf_subset), by=x->x[1])
    A64 = prob["A"]
    b64 = prob["b"]
    m, n = prob["m"], prob["n"]

    # CoreNNLS Float64 residual as baseline
    x64 = nnls(copy(A64), copy(b64))
    r64 = norm(b64 - A64 * x64) / norm(b64)

    setprecision(128) do
        ABF = BigFloat.(A64)
        bBF = BigFloat.(b64)

        # Warm up
        nnls(copy(ABF), copy(bBF))
        nonneg_lsq(copy(ABF), copy(bBF); alg=:nnls)

        tc_bf = @belapsed nnls(copy($ABF), copy($bBF))
        tn_bf = @belapsed nonneg_lsq(copy($ABF), copy($bBF); alg=:nnls)

        xC_bf = nnls(copy(ABF), copy(bBF))
        xN_bf = vec(nonneg_lsq(copy(ABF), copy(bBF); alg=:nnls))

        rC_bf = norm(bBF - ABF * xC_bf) / norm(bBF)
        rN_bf = norm(bBF - ABF * xN_bf) / norm(bBF)

        speedup_bf  = tn_bf / tc_bf
        genau_bf    = Float64(rC_bf) > 1e-60 ? Float64(rN_bf) / Float64(rC_bf) : 1.0

        sp_str  = string(round(speedup_bf, digits=2))
        gn_str  = isinf(genau_bf) ? "∞" : string(round(genau_bf, digits=2))
        rC_str  = string(round(Float64(rC_bf), sigdigits=3))
        r64_str = string(round(Float64(r64),   sigdigits=3))

        println("| $name | $m | $n | $sp_str | $gn_str | $rC_str | $r64_str |")
    end
end

println()
println("---")
println("*Benchmark abgeschlossen. Probleme aus: `benchmark/problems.jld2`*")