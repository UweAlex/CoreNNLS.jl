#test/runtests.jl
using Test
using LinearAlgebra
using CoreNNLS
using Random

@testset "CoreNNLS.jl - Forensic Validation" begin
    
    # --- Vorhandene Basis-Tests ---
    @testset "1. Interior & Boundary Basics" begin
        # Interior
        A = [1.0 0.5; 0.5 2.0]; b = [3.0, 4.0]
        @test nnls(A, b) ≈ A \ b
        
        # Boundary (einfach)
        A = [1.0 0.0; 0.0 1.0]; b = [1.0, -1.0]
        @test nnls(A, b) ≈ [1.0, 0.0]
        
        # Zero
        @test nnls(diagm(ones(2)), [-1.0, -2.0]) ≈ [0.0, 0.0] atol=1e-12
    end

    # --- Spezial-Tests für Standalone-Berechtigung ---
    
    @testset "2. High-Precision (BigFloat)" begin
        # Wir nutzen eine kleine Hilbert-Matrix (bekanntlich extrem schlecht konditioniert)
        # n=10 ist für Float64 oft schon das Ende der Präzision
        n = 10
        T = BigFloat
        setprecision(BigFloat, 256) do
            A = T[inv(T(i + j - 1)) for i in 1:n, j in 1:n]
            x_true = ones(T, n)
            b = A * x_true
            
            # Solve with Float64 (sollte scheitern oder ungenau sein)
            x_f64 = nnls(Float64.(A), Float64.(b))
            err_f64 = norm(x_f64 - ones(n))
            
            # Solve with BigFloat
            ws = NNLSWorkspace(n, n, T)
            status, x_big = nnls!(ws, A, b)
            err_big = norm(x_big - x_true)
            
            @test status == :Success
            @test err_big < err_f64
            @test err_big < 1e-30 # BigFloat muss massiv genauer sein
        end
    end

    @testset "3. Rank Deficiency & Collinearity" begin
        # Zwei identische Spalten
        A = [1.0 1.0; 1.0 1.0; 1.0 1.0]
        b = [2.0, 2.0, 2.0]
        
        # Der Solver muss hier deterministisch eine Spalte wählen (idx 1)
        # und darf nicht wegen Singularität im QR-Update crashen.
        ws = NNLSWorkspace(3, 2)
        status, x = nnls!(ws, A, b)
        
        @test status == :Success
        @test x[1] + x[2] ≈ 2.0
        @test all(x .>= 0)
        @test norm(A*x - b) < 1e-12
    end

    @testset "4. In-Place & Zero Allocation" begin
        # Testet, ob der Workspace wirklich keine neuen Allokationen im Loop macht
        # (Erfordert 'BenchmarkTools' in den Dev-Dependencies oder manuelles Zählen)
        m, n = 20, 10
        A = randn(m, n)
        b = randn(m)
        ws = NNLSWorkspace(m, n)
        
        # Erster Lauf (JIT)
        nnls!(ws, A, b)
        
        # Zweiter Lauf: Messung
        # Hinweis: In CI-Umgebungen nutzen wir oft einfache @allocated Checks
        allocs = @allocated nnls!(ws, A, b)
        
        # Da count(passive_set) und QR-Views in Julia 1.6+ sehr effizient sind, 
        # sollte das gegen 0 gehen (oder sehr klein sein).
        @test allocs < 512 # Kleine Puffer sind akzeptabel, aber kein m*n
    end

    @testset "5. Corner Case: Large Residuals" begin
        # b ist orthogonal auf Spaltenraum von A
        A = [1.0; 0.0;;] # 2x1 Matrix
        b = [0.0, 10.0]
        x = nnls(A, b)
        @test x[1] == 0.0
        @test norm(A*x - b) ≈ 10.0
    end

    @testset "6. Stress Test (Randomized)" begin
        Random.seed!(123)
        for _ in 1:50
            m, n = rand(20:100), rand(5:20)
            A = randn(m, n)
            b = randn(m)
            x = nnls(A, b)
            
            # KKT 1: Feasibility
            @test all(x .>= -1e-14)
            
            # KKT 2: Dual Feasibility (Simplified)
            # w = A'(b - Ax). Wenn x_i = 0, muss w_i <= tol sein.
            w = transpose(A) * (b - A*x)
            for j in 1:n
                if x[j] < 1e-12
                    @test w[j] <= 1e-7 # Toleranz für Float64
                else
                    @test abs(w[j]) <= 1e-7 # Stationarität im Passiv-Set
                end
            end
        end
    end
end
