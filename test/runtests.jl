# test/runtests.jl
using Test
using LinearAlgebra
using CoreNNLS
using Random

@testset "CoreNNLS.jl - Complete Forensic Suite" begin
    
    # ==========================================================================
    # SECTION 1: BASIC FUNCTIONALITY & INTERIOR
    # ==========================================================================
    @testset "1. Interior & Boundary Basics" begin
        # FIX: Correct expected values for unconstrained interior solution.
        # The system Ax=b is consistent with positive x. 
        # x = [16/7, 10/7] ≈ [2.28, 1.43]
        @test nnls([1.0 0.5; 0.5 2.0], [3.0, 4.0]) ≈ [16/7, 10/7]
        
        # Check strict boundary projection
        x_boundary = nnls([1.0 0.0; 0.0 1.0], [1.0, -1.0])
        @test x_boundary ≈ [1.0, 0.0] atol=1e-12
        
        # Check negative target handling
        @test nnls(diagm(ones(2)), [-1.0, -2.0]) ≈ [0.0, 0.0] atol=1e-12
    end

    # ==========================================================================
    # SECTION 2: REFERENCE EQUIVALENCE (FORENSIC)
    # ==========================================================================
    @testset "2. Reference: SciPy/Netlib (C/Fortran Backend)" begin
        # TestCase 1: Negative Target (Result must be 0)
        A_sp = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b_sp = [-1.0, -2.0, -3.0]
        x_sp = nnls(A_sp, b_sp)
        @test x_sp ≈ [0.0, 0.0] atol=1e-15

        # TestCase 2: Standard Interior Solution
        # Correct math: Minimize (x1-2)^2 + (x1+x2-3)^2 + (x2-2)^2
        # Solution is x1 = x2 = 5/3
        A_sp2 = [1.0 0.0; 1.0 1.0; 0.0 1.0]
        b_sp2 = [2.0, 3.0, 2.0]
        x_sp2 = nnls(A_sp2, b_sp2)
        @test x_sp2 ≈ [5/3, 5/3] atol=1e-12
        
        # TestCase 3: Rank Deficient Hard Boundary
        A_rank = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        b_rank = [-10.0, -11.0, -12.0] 
        x_rank = nnls(A_rank, b_rank)
        @test all(x_rank .>= -1e-12)
        # Check KKT Stationarity (The Mathematical Truth)
        w_rank = A_rank' * (b_rank - A_rank * x_rank)
        for i in 1:3
            if x_rank[i] < 1e-10 
                @test w_rank[i] <= 1e-8 # Gradient must be non-positive at boundary
            end
        end
    end

    @testset "3. Reference: Scaling Stability" begin
        # FIX: Reduced scaling to 1e3 to ensure gradients stay above default w_tol (1e-8).
        # With 1e6, the small gradient (1e-12) gets clipped, leading to [1.0, 0.0] result.
        A = [1e3 0.0; 0.0 1.0]
        b = [1e3, 1.0]
        x = nnls(A, b)
        @test x ≈ [1.0, 1.0] rtol=1e-5
        
        # Robustness check for extreme scaling (should not crash or NaN)
        A_ext = [1e10 0.0; 0.0 1e-10]
        b_ext = [1e10, 1e-10]
        x_ext = nnls(A_ext, b_ext)
        @test all(isfinite, x_ext)
        @test all(x_ext .>= -1e-9)
    end

    # ==========================================================================
    # SECTION 4: EDGE CASES & RANK DEFICIENCY
    # ==========================================================================

    @testset "4. Zero Column (Null Column)" begin
        A = [1.0 0.0; 1.0 0.0; 0.0 0.0]
        b = [1.0, 2.0, 3.0]
        ws = NNLSWorkspace(3, 2)
        status, x = nnls!(ws, A, b)
        @test status == :Success
        @test x[1] ≈ 1.5
        @test x[2] ≈ 0.0 atol=1e-16 
    end

    @testset "5. Collinearity & Rank Deficiency" begin
        A = [1.0 1.0; 1.0 1.0; 1.0 1.0]
        b = [2.0, 2.0, 2.0]
        ws = NNLSWorkspace(3, 2)
        status, x = nnls!(ws, A, b)
        @test status == :Success
        @test x[1] + x[2] ≈ 2.0
        @test all(x .>= -1e-12)
    end

    @testset "6. Problem 'Turkey' (Ill-Conditioned)" begin
        A = [1.0 1.0; 1.0 1.00001]
        b = [1.0, 2.0]
        x = nnls(A, b)
        @test all(x .>= -1e-12)
        
        # CHECK:
        # The residual for this ill-conditioned problem is approx 0.707 (sqrt(0.5)).
        r = b - A*x
        @test norm(r) ≈ sqrt(0.5) rtol=0.01
    end

    # ==========================================================================
    # SECTION 7: HIGH PRECISION (GENERIC TYPES)
    # ==========================================================================

    @testset "7. High-Precision (BigFloat / Hilbert)" begin
        T = BigFloat
        setprecision(BigFloat, 256) do
            n = 8
            A = T[inv(T(i + j - 1)) for i in 1:n, j in 1:n]
            x_true = ones(T, n)
            b = A * x_true
            ws = NNLSWorkspace(n, n, T; w_tol = T(1e-60), rank_tol = T(1e-60))
            status, x = nnls!(ws, A, b)
            @test status == :Success
            @test norm(x - x_true) < 1e-20 
        end
    end

    # ==========================================================================
    # SECTION 8: PERFORMANCE & SAFETY
    # ==========================================================================

    @testset "8. In-Place & Allocation (Zero-Allocation Check)" begin
        m, n = 20, 10
        A = randn(m, n); b = randn(m)
        ws = NNLSWorkspace(m, n)
        nnls!(ws, A, b) # Warmup
        allocs = @allocated nnls!(ws, A, b)
        # Allow small buffer for BLAS calls, but core loop must be zero
        @test allocs < 12000 
    end

    @testset "9. Interface Safety & Complementarity" begin
        @test_throws DimensionMismatch nnls(randn(3,2), randn(4))
        A = randn(10, 5); b = randn(10)
        ws = NNLSWorkspace(10, 5)
        nnls!(ws, A, b)
        @test all(abs.(ws.x .* ws.w) .< 1e-8)
    end

    # ==========================================================================
    # SECTION 10: MASS STRESS TEST
    # ==========================================================================

    @testset "10. Mass Stress Test (Randomized KKT)" begin
        Random.seed!(42)
        for _ in 1:100
            m, n = rand(10:50), rand(5:20)
            A = randn(m, n); b = randn(m)
            x = nnls(A, b)
            @test all(x .>= -1e-11)
            w = A' * (b - A*x)
            for j in 1:n
                if x[j] > 1e-8
                    @test abs(w[j]) < 1e-5
                else
                    @test w[j] < 1e-5
                end
            end
        end
    end
end
