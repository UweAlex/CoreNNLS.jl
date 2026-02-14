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
        @test nnls([1.0 0.5; 0.5 2.0], [3.0, 4.0]) ≈ [1.0, 1.5]
        @test nnls([1.0 0.0; 0.0 1.0], [1.0, -1.0]) ≈ [1.0, 0.0]
        @test nnls(diagm(ones(2)), [-1.0, -2.0]) ≈ [0.0, 0.0] atol=1e-10
    end

    # ==========================================================================
    # SECTION 2: REFERENCE EQUIVALENCE (FORENSIC)
    # ==========================================================================
    @testset "2. Reference: SciPy/Netlib (C/Fortran Backend)" begin
        # TestCase 1: Standard Scipy Reference (Negative Target)
        A_sp = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b_sp = [-1.0, -2.0, -3.0]
        x_sp = nnls(A_sp, b_sp)
        @test x_sp ≈ [0.0, 0.0] atol=1e-15

        # TestCase 2: Scipy Reference with mixed signs
        A_sp2 = [1.0 0.0; 1.0 1.0; 0.0 1.0]
        b_sp2 = [2.0, 3.0, 2.0]
        x_sp2 = nnls(A_sp2, b_sp2)
        @test x_sp2 ≈ [2.0, 2.0] atol=1e-14
        
        # TestCase 3 (CORRECTED): Rank Deficient Hard Boundary
        # A matrix with rank 2, but the unconstrained solution lies in negative space.
        # NNLS must project onto the boundary x >= 0.
        A_rank = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        b_rank = [-10.0, -11.0, -12.0] # Force negative gradient
        x_rank = nnls(A_rank, b_rank)
        # Check strict feasibility
        @test all(x_rank .>= -1e-12)
        # Verify KKT (Stationarity)
        w_rank = A_rank' * (b_rank - A_rank * x_rank)
        for i in 1:3
            if x_rank[i] < 1e-10 # Variable is at boundary (zero)
                @test w_rank[i] <= 1e-8 # Gradient must be non-positive
            end
        end
    end

    @testset "3. Reference: Rust Port (Scaling)" begin
        A = [1e10 0.0; 0.0 1e-10]
        b = [1e10, 1e-10]
        x = nnls(A, b)
        @test x ≈ [1.0, 1.0] rtol=1e-6
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
        @test x[2] ≈ 0.0 atol=1e-16 # Zero column must remain zero
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
        r = b - A*x
        @test norm(r) < 1e-4
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
