# test/runtests.jl
using Test
using LinearAlgebra
using CoreNNLS
using Random

@testset "CoreNNLS.jl - Forensic Validation" begin
    
    @testset "1. Interior & Boundary Basics" begin
        # Interior solution
        A = [1.0 0.5; 0.5 2.0]
        b = [3.0, 4.0]
        @test nnls(A, b) ≈ A \ b
        
        # Boundary solution
        A = [1.0 0.0; 0.0 1.0]
        b = [1.0, -1.0]
        @test nnls(A, b) ≈ [1.0, 0.0]
        
        # Zero solution
        @test nnls(diagm(ones(2)), [-1.0, -2.0]) ≈ [0.0, 0.0] atol=1e-12
    end

    @testset "2. High-Precision (BigFloat)" begin
        n = 10
        T = BigFloat
        # Nutze 256-bit Präzision für die berüchtigte Hilbert-Matrix
        setprecision(BigFloat, 256) do
            A = T[inv(T(i + j - 1)) for i in 1:n, j in 1:n]
            x_true = ones(T, n)
            b = A * x_true
            
            # WICHTIG: Die Standard-Toleranzen (1e-8) sind für BigFloat-Forensik zu grob.
            # Wir verschärfen sie massiv, damit der Solver die Präzision ausnutzt.
            ws = NNLSWorkspace(n, n, T; w_tol = T(1e-70), rank_tol = T(1e-70))
            status, x_big = nnls!(ws, A, b)
            err_big = norm(x_big - x_true)
            
            @test status == :Success
            # BigFloat muss hier um Größenordnungen besser sein als Float64 (ca. 1e-12)
            @test err_big < 1e-25 
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
        @test norm(A*x - b) < 1e-12
    end

    @testset "4. In-Place & Allocation Check" begin
        m, n = 20, 10
        A = randn(m, n)
        b = randn(m)
        ws = NNLSWorkspace(m, n)
        
        # JIT-Warmup
        nnls!(ws, A, b)
        
        # Messung der Allokationen im Loop
        allocs = @allocated nnls!(ws, A, b)
        
        # Unter Julia 1.6 allokieren QR-Faktorisierungen auf Views oft etwas Speicher.
        # 10 KB ist ein sicheres Limit, das "echte" Datenkopien (Matrix-Clones) ausschließt.
        @test allocs < 10000 
    end

    @testset "5. Corner Case: Large Residuals" begin
        # Erzeuge eine Matrix-Struktur für den Vektor
        A = reshape([1.0, 0.0], 2, 1) 
        b = [0.0, 10.0]
        
        x = nnls(A, b)
        
        @test size(x, 1) == 1
        @test x[1] == 0.0
        @test norm(A*x - b) ≈ 10.0
    end

    @testset "6. Stress Test (Randomized KKT)" begin
        Random.seed!(123)
        for _ in 1:50
            m, n = rand(20:50), rand(5:15)
            A = randn(m, n)
            b = randn(m)
            x = nnls(A, b)
            
            # KKT 1: Zulässigkeit
            @test all(x .>= -1e-13)
            
            # KKT 2: Komplementarität / Stationarität
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

    @testset "7. Advanced & Pathological Cases" begin
        
        @testset "Underdetermined (n > m)" begin
            Random.seed!(1); m, n = 20, 50
            A = randn(m, n)
            x_true = zeros(n); x_true[1:5] = rand(5)
            b = A * x_true
            
            x_nnls = nnls(A, b)
            @test norm(A*x_nnls - b) < 1e-10
            # Active-Set Theorie: Nicht mehr als m Nicht-Nullen
            @test count(x -> x > 1e-10, x_nnls) <= m 
        end

        @testset "Collinear / Ill-conditioned" begin
            eps_val = 1e-13
            A = [1.0 1.0+eps_val; 
                 1.0 1.0]
            b = [2.0, 2.0]
            
            x = nnls(A, b)
            @test norm(A*x - b) < 1e-12
        end

        @testset "KKT-Complementarity (Forensic Check)" begin
            A = randn(10, 5); b = randn(10)
            ws = NNLSWorkspace(10, 5)
            status, x = nnls!(ws, A, b)
            
            # Prüfe x_i * w_i ≈ 0 (Complementary Slackness)
            complementarity = [x[i] * ws.w[i] for i in 1:5]
            @test all(abs.(complementarity) .< 1e-10)
        end
    end
end
