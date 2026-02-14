# test/runtests.jl
using Test
using LinearAlgebra
using CoreNNLS
using Random

@testset "CoreNNLS.jl - Complete Forensic Suite" begin
    
    @testset "1. Interior & Boundary Basics" begin
        # Standard: Lösung im Inneren
        @test nnls([1.0 0.5; 0.5 2.0], [3.0, 4.0]) ≈ [1.0, 1.5] # A \ b
        # Rand: Eine Komponente wird Null
        @test nnls([1.0 0.0; 0.0 1.0], [1.0, -1.0]) ≈ [1.0, 0.0]
        # Null: Alles negativ
        @test nnls(diagm(ones(2)), [-1.0, -2.0]) ≈ [0.0, 0.0] atol=1e-12
    end

    @testset "2. High-Precision (BigFloat / Hilbert)" begin
        T = BigFloat
        setprecision(BigFloat, 256) do
            n = 10
            A = T[inv(T(i + j - 1)) for i in 1:n, j in 1:n]
            x_true = ones(T, n)
            b = A * x_true
            ws = NNLSWorkspace(n, n, T; w_tol = T(1e-70), rank_tol = T(1e-70))
            status, x = nnls!(ws, A, b)
            @test status == :Success
            @test norm(x - x_true) < 1e-25 
        end
    end

    @testset "3. Rank Deficiency & Collinearity" begin
        # Identische Spalten provozieren den Rank-Guard
        A = [1.0 1.0; 1.0 1.0; 1.0 1.0]
        b = [2.0, 2.0, 2.0]
        ws = NNLSWorkspace(3, 2)
        status, x = nnls!(ws, A, b)
        @test status == :Success
        @test x[1] + x[2] ≈ 2.0
        @test all(x .>= 0)
    end

    @testset "4. In-Place & Allocation (LTS Compliance)" begin
        m, n = 20, 10
        A = randn(m, n); b = randn(m)
        ws = NNLSWorkspace(m, n)
        nnls!(ws, A, b) # Warmup
        allocs = @allocated nnls!(ws, A, b)
        @test allocs < 10000 # Erlaubt Overhead für Julia 1.6
    end

    @testset "5. Edge Case Dimensions" begin
        # 1x1 Matrix
        @test nnls([2.0;;], [4.0]) ≈ [2.0]
        # Underdetermined 1xN
        A_1n = [1.0 2.0 3.0]; b_1n = [6.0]
        x = nnls(A_1n, b_1n)
        @test A_1n * x ≈ b_1n
        @test count(xi -> xi > 1e-10, x) == 1
    end

    @testset "6. Pathological Loops (Backtracking)" begin
        # Erzwingt das Löschen einer Variable aus dem Passiv-Set (Phase B)
        # Dies ist der am schwersten zu testende Zustand des Lawson-Hanson Algorithmus
        A = [1.0 -1.0; 1.0 2.0; 2.0 1.0]
        b = [-1.0, 1.0, 1.0]
        x = nnls(A, b)
        @test all(x .>= 0)
        # Check KKT Stationarity
        w = A' * (b - A*x)
        @test all(w .<= 1e-12)
    end

    @testset "7. Mass Stress Test (KKT)" begin
        Random.seed!(42)
        for _ in 1:100
            m, n = rand(1:50), rand(1:30)
            A = randn(m, n); b = randn(m)
            x = nnls(A, b)
            w = A' * (b - A*x)
            @test all(x .>= -1e-12)
            for j in 1:n
                if x[j] > 1e-9
                    @test abs(w[j]) < 1e-7 # Aktiv: Gradient muss 0 sein
                else
                    @test w[j] < 1e-7 # Passiv: Darf keine Abstiegsrichtung sein
                end
            end
        end
    end

    @testset "8. Industry Benchmarks (Lawson-Hanson)" begin
        # Original-Daten von 1974
        A = [0.03 0.01; 0.35 0.01; 0.33 0.02; 0.90 0.01]
        b = [1.0, 2.0, 3.0, 4.0]
        @test nnls(A, b) ≈ [3.491331, 0.0] rtol=1e-5
    end

    @testset "9. Tikhonov Regularization (Ridge)" begin
        # Stabilisiert das Problem, typischer Python/SciPy Use-Case
        A = [1.0 2.0; 3.0 4.0]; b = [1.0, 2.0]; λ = 0.5
        A_reg = [A; λ * I]
        b_reg = [b; zeros(2)]
        x = nnls(A_reg, b_reg)
        @test all(x .>= 0)
    end

    @testset "10. Interface Safety & Complementarity" begin
        # Dimension Mismatch
        @test_throws DimensionMismatch nnls(randn(3,2), randn(4))
        # Complementarity Check (x_i * w_i = 0)
        A = randn(10, 5); b = randn(10)
        ws = NNLSWorkspace(10, 5)
        nnls!(ws, A, b)
        @test all(abs.(ws.x .* ws.w) .< 1e-10)
    end
end
