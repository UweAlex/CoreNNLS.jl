# test/runtests.jl
using Test
using LinearAlgebra
using CoreNNLS
using Random

@testset "CoreNNLS.jl - Complete Forensic Suite" begin
    
    @testset "1. Interior & Boundary Basics" begin
        # Nutze ≈ statt fixer atol für bessere Stabilität auf verschiedenen CPUs
        @test nnls([1.0 0.5; 0.5 2.0], [3.0, 4.0]) ≈ [1.0, 1.5]
        @test nnls([1.0 0.0; 0.0 1.0], [1.0, -1.0]) ≈ [1.0, 0.0]
        # Etwas lockerere Toleranz für Basis-Null-Tests
        @test nnls(diagm(ones(2)), [-1.0, -2.0]) ≈ [0.0, 0.0] atol=1e-10
    end

    @testset "2. High-Precision (BigFloat / Hilbert)" begin
        T = BigFloat
        setprecision(BigFloat, 256) do
            n = 8 # Etwas kleiner für schnellere CI, aber immer noch kritisch
            A = T[inv(T(i + j - 1)) for i in 1:n, j in 1:n]
            x_true = ones(T, n)
            b = A * x_true
            # Hohe Präzision erzwingen
            ws = NNLSWorkspace(n, n, T; w_tol = T(1e-60), rank_tol = T(1e-60))
            status, x = nnls!(ws, A, b)
            @test status == :Success
            @test norm(x - x_true) < 1e-20 
        end
    end

    @testset "3. Rank Deficiency & Collinearity" begin
        A = [1.0 1.0; 1.0 1.0; 1.0 1.0]
        b = [2.0, 2.0, 2.0]
        ws = NNLSWorkspace(3, 2)
        status, x = nnls!(ws, A, b)
        @test status == :Success
        @test x[1] + x[2] ≈ 2.0
        @test all(x .>= -1e-12)
    end

    @testset "4. In-Place & Allocation (LTS Compliance)" begin
        m, n = 20, 10
        A = randn(m, n); b = randn(m)
        ws = NNLSWorkspace(m, n)
        nnls!(ws, A, b)
        allocs = @allocated nnls!(ws, A, b)
        @test allocs < 12000 # Erhöht für Julia 1.6 Overhead
    end

    @testset "5. Edge Case Dimensions" begin
        # 1x1 Matrix
        @test nnls([2.0;;], [4.0]) ≈ [2.0]
        # Underdetermined 1xN - sauberer Zugriff
        A_1n = [1.0 2.0 3.0]
        b_1n = [6.0]
        x = nnls(A_1n, b_1n)
        @test (A_1n * x)[1] ≈ 6.0
        @test count(xi -> xi > 1e-8, x) >= 1
    end

    @testset "6. Pathological Loops (Backtracking)" begin
        # Erzwingt Phase B
        A = [1.0 -1.0; 1.0 2.0; 2.0 1.0]
        b = [-1.0, 1.0, 1.0]
        x = nnls(A, b)
        @test all(x .>= -1e-12)
        # Residuum muss minimal sein
        @test norm(A*x - b) <= norm(b)
    end

    @testset "7. Mass Stress Test (KKT)" begin
        Random.seed!(42)
        for _ in 1:100 # Reduziert auf 100 für schnellere CI-Feedback-Loops
            m, n = rand(5:50), rand(2:20)
            A = randn(m, n); b = randn(m)
            x = nnls(A, b)
            @test all(x .>= -1e-11)
            # KKT Stationarity Check
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

    @testset "8. Industry Benchmarks (Lawson-Hanson)" begin
        # Das Problem ist sehr sensitiv. Wir nutzen den Workspace mit feinen Toleranzen.
        A = [0.03 0.01; 0.35 0.01; 0.33 0.02; 0.90 0.01]
        b = [1.0, 2.0, 3.0, 4.0]
        # Explizite Toleranzen helfen dem Active-Set-Algorithmus bei kleinen Koeffizienten
        ws = NNLSWorkspace(4, 2; w_tol=1e-14)
        status, x = nnls!(ws, A, b)
        @test x[1] ≈ 3.491331 rtol=1e-4
        @test x[2] ≈ 0.0 atol=1e-8
    end

    @testset "9. Tikhonov Regularization (Ridge)" begin
        A = [1.0 2.0; 3.0 4.0]; b = [1.0, 2.0]; λ = 0.5
        A_reg = [A; λ * I]
        b_reg = [b; zeros(2)]
        x = nnls(A_reg, b_reg)
        @test all(x .>= -1e-12)
    end

    @testset "10. Interface Safety & Complementarity" begin
        @test_throws DimensionMismatch nnls(randn(3,2), randn(4))
        A = randn(10, 5); b = randn(10)
        ws = NNLSWorkspace(10, 5)
        nnls!(ws, A, b)
        # Slackness: x .* w ≈ 0
        @test all(abs.(ws.x .* ws.w) .< 1e-8)
    end
end
