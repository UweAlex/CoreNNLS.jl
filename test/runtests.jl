# test/runtests.jl
using Test
using LinearAlgebra
using CoreNNLS
using Random

@testset "CoreNNLS.jl - Forensic Validation" begin
    
    @testset "1. Interior & Boundary Basics" begin
        # Interior
        A = [1.0 0.5; 0.5 2.0]
        b = [3.0, 4.0]
        @test nnls(A, b) ≈ A \ b
        
        # Boundary
        A = [1.0 0.0; 0.0 1.0]
        b = [1.0, -1.0]
        @test nnls(A, b) ≈ [1.0, 0.0]
        
        # Zero Solution
        @test nnls(diagm(ones(2)), [-1.0, -2.0]) ≈ [0.0, 0.0] atol=1e-12
    end

    @testset "2. High-Precision (BigFloat)" begin
        n = 10
        T = BigFloat
        # Nutze einen Block, um Präzision sicherzustellen
        setprecision(BigFloat, 256) do
            # Hilbert Matrix: Inv(i+j-1)
            A = T[inv(T(i + j - 1)) for i in 1:n, j in 1:n]
            x_true = ones(T, n)
            b = A * x_true
            
            # Vergleich mit Float64
            x_f64 = nnls(Float64.(A), Float64.(b))
            err_f64 = norm(x_f64 - ones(n))
            
            # Solve with CoreNNLS BigFloat
            ws = NNLSWorkspace(n, n, T)
            status, x_big = nnls!(ws, A, b)
            err_big = norm(x_big - x_true)
            
            @test status == :Success
            @test err_big < err_f64
            @test err_big < 1e-25 
        end
    end

    @testset "3. Rank Deficiency & Collinearity" begin
        # Zwei identische Spalten (Singularität)
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
        
        # Messung: In Julia 1.6 können Views minimal allokieren
        allocs = @allocated nnls!(ws, A, b)
        
        # Ziel: Nahezu null Allokationen (Toleranz für CI Overhead)
        @test allocs < 1024 
    end

    @testset "5. Corner Case: Large Residuals" begin
        # Fix: Nutze reshape oder [;,,], um sicherzustellen, dass A eine Matrix ist
        A = reshape([1.0, 0.0], 2, 1) 
        b = [0.0, 10.0]
        
        # Das löst das Problem: min || [1;0]*x - [0;10] || 
        # Da x >= 0, ist x=0 die beste Lösung
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
            
            # KKT 1: Feasibility
            @test all(x .>= -1e-13)
            
            # KKT 2: Stationarity & Dual Feasibility
            # w = A'(b - Ax)
            w = transpose(A) * (b - A*x)
            for j in 1:n
                if x[j] < 1e-10
                    # Falls x=0, darf keine Abstiegsrichtung existieren (w <= 0)
                    @test w[j] <= 1e-7 
                else
                    # Falls x>0, muss der Gradient Null sein
                    @test abs(w[j]) <= 1e-7 
                end
            end
        end
    end
end
