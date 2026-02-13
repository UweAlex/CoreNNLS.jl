using Test
using LinearAlgebra
using CoreNNLS
using Random # Wird jetzt über Project.toml verwaltet

@testset "CoreNNLS.jl - Validation" begin
    
    @testset "1. Interior Solution" begin
        A = [1.0 0.5; 0.5 2.0]
        b = [3.0, 4.0]
        x_exact = A \ b
        
        ws = NNLSWorkspace(2, 2)
        status, x_nnls = nnls!(ws, A, b)
        
        @test status == :Success
        @test x_nnls ≈ x_exact rtol=1e-6
    end

    @testset "2. Boundary Solution" begin
        A = [1.0 0.0; 0.0 1.0]
        b = [1.0, -1.0]
        x_exact = [1.0, 0.0]
        
        x_nnls = nnls(A, b)
        @test x_nnls ≈ x_exact rtol=1e-6
    end

    @testset "3. Rank Deficiency" begin
        A = [1.0 2.0; 1.0 2.0] 
        b = [3.0, 3.0] 
        
        x_nnls = nnls(A, b)
        residual = norm(A * x_nnls - b)
        
        @test residual < 1e-6
        @test all(x_nnls .>= -1e-10)
    end
    
    @testset "4. Zero Solution" begin
        A = [1.0 0.0; 0.0 1.0]
        b = [-1.0, -1.0]
        
        x_nnls = nnls(A, b)
        @test x_nnls ≈ [0.0, 0.0] atol=1e-10
    end

    @testset "5. Stress Test (100 Random Matrices)" begin
        Random.seed!(42) # Deterministischer Zufall
        
        for i in 1:100
            m = rand(10:50)
            n = rand(5:20)
            
            A = randn(m, n)
            b = randn(m)
            
            ws = NNLSWorkspace(m, n)
            status, x_core = nnls!(ws, A, b)
            
            @test all(isfinite, x_core)
            @test all(x_core .>= -1e-10)
            
            r_sol = norm(A * x_core - b)
            r0 = norm(b) # Residuum bei x=0
            @test r_sol <= r0 + 1e-8
        end
    end
end
