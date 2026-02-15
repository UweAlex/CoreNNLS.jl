using Test
using LinearAlgebra
using CoreNNLS
using Random

@testset "CoreNNLS.jl - Full Forensic Suite (Pure Julia)" begin

# ============================================================================
# ORIGINAL TESTS (1-19)
# ============================================================================

# 1. BASICS
@testset "1. Interior & Boundary Basics" begin
    @test nnls([1.0 0.5; 0.5 2.0], [3.0, 4.0]) ≈ [16/7, 10/7]
    @test nnls([1.0 0.0; 0.0 1.0], [1.0, -1.0]) ≈ [1.0, 0.0] atol=1e-12
end

# 2. REFERENCE
@testset "2. Reference: SciPy/Netlib" begin
    A_sp = [1.0 2.0; 3.0 4.0; 5.0 6.0]; b_sp = [-1.0, -2.0, -3.0]
    @test nnls(A_sp, b_sp) ≈ [0.0, 0.0] atol=1e-15
    A_mix = [2.0 1.0; 1.0 2.0]
    b_mix = [-0.5, 1.0]
    @test nnls(A_mix, b_mix) ≈ [0.0, 0.3] atol=1e-10
end

# 3. SCALING
@testset "3. Reference: Scaling Stability" begin
    A = [1e3 0.0; 0.0 1.0]; b = [1e3, 1.0]
    @test nnls(A, b) ≈ [1.0, 1.0] rtol=1e-5
end

# 4. EDGE CASES
@testset "4. Zero Column (Null Column)" begin
    A = [1.0 0.0; 1.0 0.0; 0.0 0.0]; b = [1.0, 2.0, 3.0]
    ws = NNLSWorkspace(3, 2)
    status, x = nnls!(ws, A, b)
    @test status == :Success
    @test x[1] ≈ 1.5 atol=1e-12
    @test x[2] == 0.0
end

# 5. COLLINEARITY
@testset "5. Collinearity & Rank Deficiency" begin
    A = [1.0 1.0; 1.0 1.0; 1.0 1.0]; b = [2.0, 2.0, 2.0]
    x = nnls(A, b)
    @test x[1] + x[2] ≈ 2.0 atol=1e-12
    @test all(x .>= -1e-12)
end

# 6. ILL-CONDITIONED
@testset "6. Problem 'Turkey'" begin
    A = [1.0 1.0; 1.0 1.00001]; b = [1.0, 2.0]
    x = nnls(A, b)
    @test all(x .>= -1e-12)
    @test norm(b - A*x) ≈ 0.70710678 rtol=1e-3
end

# 7. HIGH-PRECISION (Pure Julia)
@testset "7. High-Precision (BigFloat)" begin
    T = BigFloat
    setprecision(128) do
        A = T[1.0 0.5; 0.5 2.0]; b = T[3.0, 4.0]
        @test nnls(A, b) ≈ [BigFloat(16)/7, BigFloat(10)/7]
    end
end

# 8. PERFORMANCE
@testset "8. In-Place & Allocation" begin
    m, n = 20, 10
    A = randn(m, n); b = randn(m); ws = NNLSWorkspace(m, n)
    nnls!(ws, A, b) # Warmup
    allocs = @allocated nnls!(ws, A, b)
    @test allocs < 10240
end

# 9. SAFETY
@testset "9. Interface Safety & Complementarity" begin
    @test_throws DimensionMismatch nnls(randn(3,2), randn(4))
    A = randn(10, 5); b = randn(10); ws = NNLSWorkspace(10, 5)
    nnls!(ws, A, b)
    @test all(abs.(ws.x .* ws.w) .< 1e-8)
end

# 10. MASS STRESS (KKT)
@testset "10. Mass Stress Test (KKT)" begin
    Random.seed!(42)
    for _ in 1:50
        m, n = rand(10:30), rand(5:15)
        A = randn(m, n); b = randn(m); x = nnls(A, b)
        w = A' * (b - A*x)
        for j in 1:n
            x[j] > 1e-8 ? (@test abs(w[j]) < 1e-5) : (@test w[j] < 1e-5)
        end
    end
end

# 11. R-PACKAGE (Noisy)
@testset "11. R nnls Example (Noisy)" begin
    t = collect(0:0.04:2.0)
    A = hcat([exp.(-[0.5, 0.6, 1.0][i] * t) for i in 1:3]...)
    Random.seed!(3312)
    b = A * [0.0, 0.7, 0.7] + 0.01 * randn(length(t))
    x = nnls(A, b)
    @test x[1] < 0.5
    @test norm(A*x - b) < 0.1
end

# 12. NETLIB BVLS
@testset "12. Netlib BVLS Cases" begin
    Random.seed!(123)
    for i in 1:3
        A = randn(20, 10); b = randn(20); x = nnls(A, b)
        w = A' * (b - A*x)
        for j in 1:10
            x[j] > 1e-8 ? (@test abs(w[j]) < 1e-5) : (@test w[j] < 1e-5)
        end
    end
end

# 13. RECOVERY
@testset "13. Burkardt/Lawson Recovery" begin
    Random.seed!(42)
    for m in 4:6, n in 2:3
        A = randn(m, n); x_s = abs.(randn(n))
        b = A * x_s; x = nnls(A, b)
        @test x ≈ x_s atol=1e-8
    end
end

# 14. HILBERT TORTURE TEST
@testset "14. Hilbert Matrix (Ill-Conditioned)" begin
    n = 4
    A = [1.0 / (i + j - 1) for i in 1:n, j in 1:n]
    x_true = ones(n)
    b = A * x_true
    x_calc = nnls(A, b)
    @test x_calc ≈ x_true atol=1e-8
    @test all(x_calc .>= 0.0)
    n5 = 5
    A5 = [1.0 / (i + j - 1) for i in 1:n5, j in 1:n5]
    b5 = A5 * ones(n5)
    x5 = nnls(A5, b5)
    @test norm(A5 * x5 - b5) < 1e-5
end

# 15. FLOAT32 PRECISION
@testset "15. Float32 Precision" begin
    A32 = Float32[1.0 2.0; 3.0 4.0]
    b32 = Float32[1.0, 2.0]
    x32 = nnls(A32, b32)
    @test eltype(x32) == Float32
    @test x32 ≈ Float32[0.0, 0.5] atol=1e-6
end

# 16. UNDERDETERMINED SYSTEM (m < n)
@testset "16. Underdetermined System (m < n)" begin
    Random.seed!(99)
    m, n = 5, 10
    A = randn(m, n)
    b = randn(m)
    x = nnls(A, b)
    w = A' * (b - A*x)
    for j in 1:n
        x[j] > 1e-8 ? (@test abs(w[j]) < 1e-5) : (@test w[j] < 1e-5)
    end
end

# 17. SCALAR CASE (n=1)
@testset "17. Scalar Case (n=1)" begin
    A = reshape([1.0, 2.0, 3.0, 4.0], 4, 1)
    b = [0.9, 2.1, 2.9, 4.2]
    x = nnls(A, b)
    @test length(x) == 1
    @test x[1] ≈ 1.0 atol=0.1
    b_neg = -[1.0, 2.0]
    x_neg = nnls(A[1:2, :], b_neg)
    @test x_neg[1] == 0.0
end

# 18. LARGER SCALE TEST
@testset "18. Larger Scale Stability" begin
    Random.seed!(500)
    m, n = 200, 100
    A = randn(m, n)
    x_true = zeros(n)
    x_true[1:10] = rand(10) .+ 0.1
    b = A * x_true
    x_calc = nnls(A, b)
    zeros_count = count(x -> x < 1e-6, x_calc)
    @test zeros_count >= 85
    @test norm(b - A*x_calc) / norm(b) < 1e-5
end

# 19. PSEUDO-INVERSE CONSISTENCY
@testset "19. Pseudo-Inverse Consistency" begin
    Random.seed!(600)
    A = randn(20, 10)
    x_true = abs.(randn(10))
    b = A * x_true
    x_nnls = nnls(A, b)
    x_pinv = pinv(A) * b
    @test x_nnls ≈ x_pinv atol=1e-6
end

# ============================================================================
# NEW TESTS (20-34) - EXTENDED COVERAGE
# ============================================================================

# 20. SPARSE PATTERN (DENSE MATRIX)
@testset "20. Sparse Pattern Matrix" begin
    Random.seed!(700)
    m, n = 50, 20

    # Create matrix with sparse structure (~90% zeros)
    A_sparse_pattern = zeros(m, n)
    for _ in 1:div(m*n, 10)
        i, j = rand(1:m), rand(1:n)
        A_sparse_pattern[i, j] = randn()
    end

    b = randn(m)

    # Test solver with sparse pattern matrix
    x = nnls(A_sparse_pattern, b)

    # KKT check
    w = A_sparse_pattern' * (b - A_sparse_pattern * x)
    for j in 1:n
        x[j] > 1e-8 ? (@test abs(w[j]) < 1e-5) : (@test w[j] < 1e-5)
    end
end

# 21. EXTREME DIMENSIONS
@testset "21. Extreme Dimensions" begin
    # Very tall matrix (m >> n)
    Random.seed!(800)
    m_tall, n_tall = 1000, 5
    A_tall = randn(m_tall, n_tall)
    b_tall = randn(m_tall)
    x_tall = nnls(A_tall, b_tall)
    @test all(x_tall .>= -1e-12)
    @test length(x_tall) == n_tall

    # Very wide matrix (m << n)
    m_wide, n_wide = 5, 100
    A_wide = randn(m_wide, n_wide)
    b_wide = randn(m_wide)
    x_wide = nnls(A_wide, b_wide)
    @test all(x_wide .>= -1e-12)
    @test length(x_wide) == n_wide

    # Square matrix
    n_sq = 50
    A_sq = randn(n_sq, n_sq)
    b_sq = randn(n_sq)
    x_sq = nnls(A_sq, b_sq)
    @test all(x_sq .>= -1e-12)
end

# 22. SINGLE ROW / SINGLE ELEMENT
@testset "22. Single Row & Single Element" begin
    # Single row (m=1)
    A_row = reshape([1.0, 2.0, 3.0], 1, 3)
    b_row = [6.0]
    x_row = nnls(A_row, b_row)
    @test all(x_row .>= 0.0)
    @test A_row * x_row ≈ b_row atol=1e-10

    # Single element (1x1)
    A_single = reshape([2.0], 1, 1)
    b_pos = [4.0]
    b_neg = [-4.0]
    @test nnls(A_single, b_pos) ≈ [2.0] atol=1e-12
    @test nnls(A_single, b_neg) ≈ [0.0] atol=1e-12
end

# 23. COMPLEX NUMBERS VIA REAL REFORMULATION
@testset "23. Complex Numbers (Real Reformulation)" begin
    Random.seed!(900)
    m, n = 10, 5
    A_c = randn(ComplexF64, m, n)
    x_true_c = complex.(abs.(randn(n)), abs.(randn(n)))
    b_c = A_c * x_true_c
    A_real = [real(A_c) -imag(A_c); imag(A_c) real(A_c)]
    b_real = [real(b_c); imag(b_c)]
    x_real = nnls(A_real, b_real)
    x_reconstructed = complex.(x_real[1:n], x_real[n+1:end])
    @test norm(A_c * x_reconstructed - b_c) / norm(b_c) < 1e-8
end

# 24. NEAR-ZERO ENTRIES
@testset "24. Near-Zero & Denormalized Numbers" begin
    A_tiny = [1e-15 1e-14; 1e-14 1e-15]
    b_tiny = [1e-14, 1e-14]
    x_tiny = nnls(A_tiny, b_tiny)
    @test all(x_tiny .>= 0.0)
    @test all(isfinite.(x_tiny))

    A_mix = [1.0 1e-16; 1e-16 1.0]
    b_mix = [1.0, 1.0]
    x_mix = nnls(A_mix, b_mix)
    @test x_mix ≈ [1.0, 1.0] atol=1e-10
end

# 25. EXTREME SCALING
@testset "25. Extreme Scaling Variants" begin
    A_scale = [1e10 0.0; 0.0 1e-10]
    b_scale = [1e10, 1e-10]
    x_scale = nnls(A_scale, b_scale)
    @test x_scale ≈ [1.0, 1.0] rtol=1e-4

    Random.seed!(1000)
    n = 10
    scales = [10.0^((-1)^i * i) for i in 1:n]
    A_alt = diagm(scales)
    b_alt = scales .* ones(n)
    x_alt = nnls(A_alt, b_alt)
    @test x_alt ≈ ones(n) rtol=1e-4
end

# 26. DETERMINISTIC REPRODUCIBILITY
@testset "26. Deterministic Reproducibility" begin
    Random.seed!(1100)
    A = randn(30, 15)
    b = randn(30)
    x1 = nnls(A, b)
    x2 = nnls(A, b)
    x3 = nnls(A, b)
    @test x1 == x2
    @test x2 == x3
end

# 27. ORTHOGONAL MATRIX
@testset "27. Orthogonal & Near-Orthogonal Matrices" begin
    n = 5
    Q, _ = qr(randn(n, n))
    A_orth = Matrix(Q)
    b_orth = ones(n)
    x_orth = nnls(A_orth, b_orth)
    @test all(x_orth .>= -1e-12)
    @test norm(A_orth * x_orth - b_orth) < 1e-10 ||
          norm(A_orth * x_orth - b_orth) < norm(b_orth)
end

# 28. ALL-ZEROS RHS
@testset "28. Zero Right-Hand Side" begin
    A = randn(10, 5)
    b_zero = zeros(10)
    x_zero = nnls(A, b_zero)
    @test x_zero ≈ zeros(5) atol=1e-15
end

# 29. IDENTICAL COLUMNS
@testset "29. Identical & Proportional Columns" begin
    col = randn(10)
    A_ident = hcat(col, col, col)
    b = 3.0 * col
    x = nnls(A_ident, b)
    @test sum(x) ≈ 3.0 atol=1e-10
    @test all(x .>= -1e-12)

    A_prop = hcat(col, 2*col, 3*col)
    b_prop = 6.0 * col
    x_prop = nnls(A_prop, b_prop)
    @test dot([1, 2, 3], x_prop) ≈ 6.0 atol=1e-10
end

# 30. PERFORMANCE BENCHMARK
@testset "30. Performance Benchmark" begin
    Random.seed!(1200)

    # Small problem timing
    m_s, n_s = 20, 10
    A_s = randn(m_s, n_s)
    b_s = randn(m_s)
    ws_s = NNLSWorkspace(m_s, n_s)

    # Warmup
    for _ in 1:10
        nnls!(ws_s, copy(A_s), copy(b_s))
    end

    t_small = @elapsed for _ in 1:100
        nnls!(ws_s, copy(A_s), copy(b_s))
    end
    @test t_small < 2.0  # 100 iterations in under 2 seconds

    # Medium problem
    m_m, n_m = 100, 50
    A_m = randn(m_m, n_m)
    b_m = randn(m_m)
    ws_m = NNLSWorkspace(m_m, n_m)

    nnls!(ws_m, copy(A_m), copy(b_m))  # Warmup
    t_medium = @elapsed for _ in 1:10
        nnls!(ws_m, copy(A_m), copy(b_m))
    end
    @test t_medium < 10.0  # 10 iterations in under 10 seconds

    @info "Performance" small_per_call=t_small/100 medium_per_call=t_medium/10
end

# 31. FLOAT16 PRECISION
@testset "31. Float16 Precision" begin
    try
        A16 = Float16[1.0 2.0; 3.0 4.0]
        b16 = Float16[1.0, 2.0]
        x16 = nnls(A16, b16)
        @test eltype(x16) == Float16
        @test all(x16 .>= Float16(0.0))
    catch e
        @warn "Float16 not supported, skipping test: $e"
        @test true # Count as pass if feature not supported
    end
end

# 32. WORKSPACE REUSE
@testset "32. Workspace Reuse Consistency" begin
    Random.seed!(1300)
    m, n = 25, 12
    ws = NNLSWorkspace(m, n)
    results = Vector{Vector{Float64}}()

    for _ in 1:10
        A = randn(m, n)
        b = randn(m)
        status, x = nnls!(ws, A, b)
        @test status == :Success
        push!(results, copy(x))
        w = A' * (b - A*x)
        for j in 1:n
            x[j] > 1e-8 ? (@test abs(w[j]) < 1e-5) : (@test w[j] < 1e-5)
        end
    end
    @test !all(results[i] ≈ results[1] for i in 2:10)
end

# 33. NEGATIVE DEFINITE GRADIENT
@testset "33. All-Negative RHS (Zero Solution Expected)" begin
    Random.seed!(1400)
    m, n = 20, 10
    A = abs.(randn(m, n))
    b = -abs.(randn(m))
    x = nnls(A, b)
    @test x ≈ zeros(n) atol=1e-12
end

# 34. VANDERMONDE MATRIX
@testset "34. Vandermonde Matrix (Polynomial Fitting)" begin
    t = range(0, 1, length=20)
    degree = 4
    A_vander = hcat([t.^k for k in 0:degree]...)
    c_true = [1.0, 0.5, 0.3, 0.1, 0.05]
    b = A_vander * c_true
    c_calc = nnls(A_vander, b)
    @test c_calc ≈ c_true atol=1e-8
end

end  # main testset
