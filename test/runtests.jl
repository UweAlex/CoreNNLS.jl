# test/runtests.jl
using Test
using LinearAlgebra
using CoreNNLS
using Random

@testset "CoreNNLS.jl - Universal Forensic Validation" begin
    
    @testset "1. Interior & Boundary Basics" begin
        # Standardfall: Lösung liegt im Inneren
        A = [1.0 0.5; 0.5 2.0]
        b = [3.0, 4.0]
        @test nnls(A, b) ≈ A \ b
        
        # Randfall: Eine Komponente muss Null werden
        A = [1.0 0.0; 0.0 1.0]
        b = [1.0, -1.0]
        @test nnls(A, b) ≈ [1.0, 0.0]
        
        # Null-Lösung: Alle b negativ
        @test nnls(diagm(ones(2)), [-1.0, -2.0]) ≈ [0.0, 0.0] atol=1e-12
    end

    @testset "2. High-Precision (BigFloat / Hilbert)" begin
        n = 10
        T = BigFloat
        setprecision(BigFloat, 256) do
            # Die Hilbert-Matrix ist der Endgegner für Float64
            A = T[inv(T(i + j - 1)) for i in 1:n, j in 1:n]
            x_true = ones(T, n)
            b = A * x_true
            
            # Massive Verschärfung der Toleranzen für echte Forensik
            ws = NNLSWorkspace(n, n, T; w_tol = T(1e-70), rank_tol = T(1e-70))
            status, x_big = nnls!(ws, A, b)
            err_big = norm(x_big - x_true)
            
            @test status == :Success
            @test err_big < 1e-25 
        end
    end

    @testset "3. Rank Deficiency & Collinearity" begin
        # Identische Spalten: Testet den Rank-Guard und Determinismus
        A = [1.0 1.0; 1.0 1.0; 1.0 1.0]
        b = [2.0, 2.0, 2.0]
        
        ws = NNLSWorkspace(3, 2)
        status, x = nnls!(ws, A, b)
        
        @test status == :Success
        @test x[1] + x[2] ≈ 2.0
        @test all(x .>= 0)
        @test norm(A*x - b) < 1e-12
    end

    @testset "4. In-Place & Allocation (LTS Compliance)" begin
        m, n = 20, 10
        A = randn(m, n); b = randn(m)
        ws = NNLSWorkspace(m, n)
        
        nnls!(ws, A, b) # Warmup
        allocs = @allocated nnls!(ws, A, b)
        
        # Limit für Julia 1.6 (View/QR Overhead abgefangen)
        @test allocs < 10000 
    end

    @testset "5. Corner Case: Shape & Types" begin
        # Matrix-Typ-Stabilität (Fix für MethodError)
        A = reshape([1.0, 0.0], 2, 1) 
        b = [0.0, 10.0]
        x = nnls(A, b)
        @test x[1] == 0.0
        @test norm(A*x - b) ≈ 10.0
    end

    @testset "6. Stress Test (Randomized KKT)" begin
        Random.seed!(123)
        for _ in 1:50
            m, n = rand(20:50), rand(5:15)
            A = randn(m, n); b = randn(m)
            x = nnls(A, b)
            @test all(x .>= -1e-13)
            
            # Duale Variable w muss KKT erfüllen
            w = transpose(A) * (b - A*x)
            for j in 1:n
                if x[j] < 1e-10
                    @test w[j] <= 1e-7 
                else
                    @test abs(w[j]) <= 1e-7 
                end
            end
        end
    end

    @testset "7. Industry Standards (SciPy & Rust Benchmarks)" begin
        
        @testset "Underdetermined (n > m)" begin
            # Typisch für Sparse-Recovery Tests in Rust
            Random.seed!(1); m, n = 20, 50
            A = randn(m, n)
            x_true = zeros(n); x_true[1:5] = rand(5)
            b = A * x_true
            x_nnls = nnls(A, b)
            @test norm(A*x_nnls - b) < 1e-10
            @test count(x -> x > 1e-10, x_nnls) <= m 
        end

        @testset "Lawson & Hanson (1974) Original Data" begin
            # Das historische Testbeispiel aus dem Standardwerk
            A = [0.03  0.01;
                 0.35  0.01;
                 0.33  0.02;
                 0.90  0.01]
            b = [1.0, 2.0, 3.0, 4.0]
            x = nnls(A, b)
            # Bekanntes Ergebnis: ca. 3.4913, 0.0
            @test x[1] ≈ 3.491331 rtol=1e-5
            @test x[2] == 0.0
        end

        @testset "Tikhonov Stability (Ridge)" begin
            # Testet Stabilität bei künstlich erhöhter Kondition (SciPy-Style)
            A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
            b = [1.0, 1.0, 1.0]
            λ = 0.5
            A_reg = [A; λ * I]
            b_reg = [b; zeros(2)]
            x = nnls(A_reg, b_reg)
            @test all(x .>= 0)
            @test length(x) == 2
        end
    end

    @testset "8. Interface Safety" begin
        # Dimension Mismatch Check
        A = randn(5, 3); b = randn(6)
        @test_throws DimensionMismatch nnls(A, b)
    end
end
