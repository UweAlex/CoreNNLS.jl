using Test
using LinearAlgebra
using CoreNNLS

@testset "CoreNNLS.jl - Analytical Validation" begin
    
    @testset "1. Interior Solution (Happy Path)" begin
        # A ist Positiv definit, b ist positiv.
        # Erwartung: x ≈ A \ b (da constraints inaktiv)
        
        A = [1.0 0.5; 
             0.5 2.0]
        b = [3.0, 4.0]
        
        x_exact = A \ b
        
        # Test Out-of-Place
        x_nnls = nnls(A, b)
        
        @test x_nnls ≈ x_exact rtol=1e-6
        @test all(x_nnls .>= 0.0)
        
        # Test In-Place 
        ws = NNLSWorkspace(2, 2)
        status, x_inplace = nnls!(ws, A, b)
        
        @test status == :Success
        @test x_inplace ≈ x_exact rtol=1e-6
    end

    @testset "2. Boundary Solution (Active Constraints)" begin
        # A = I (Identität).
        # b = [1, -1].
        # Unconstrained LS: x = [1, -1].
        # NNLS Constraint: x_2 >= 0.
        # Erwartung: x = [1, 0].
        
        A = [1.0 0.0; 
             0.0 1.0]
        b = [1.0, -1.0]
        
        x_exact = [1.0, 0.0]
        
        x_nnls = nnls(A, b)
        
        @test x_nnls ≈ x_exact rtol=1e-6
        @test x_nnls[2] ≈ 0.0 atol=1e-10
    end

    @testset "3. Rank Deficiency (Degenerierte Matrix)" begin
        # A ist singulär (Spalte 2 = 2 * Spalte 1)
        
        A = [1.0 2.0; 
             1.0 2.0] 
        b = [3.0, 3.0] 
        
        x_nnls = nnls(A, b)
        
        # Prüfen: Ist das Problem lösbar ohne Absturz?
        residual = norm(A * x_nnls - b)
        @test residual < 1e-6
        @test all(x_nnls .>= -1e-10)
    end
    
    @testset "4. Zero Solution" begin
        # b liegt vollständig außerhalb des positiven Kegels.
        
        A = [1.0 0.0; 
             0.0 1.0]
        b = [-1.0, -1.0]
        
        x_nnls = nnls(A, b)
        @test x_nnls ≈ [0.0, 0.0] atol=1e-10
    end

end
