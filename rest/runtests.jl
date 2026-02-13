using Test
using LinearAlgebra
using CoreNNLS

@testset "CoreNNLS.jl - Analytical Validation" begin
    
    @testset "1. Interior Solution (Happy Path)" begin
        # Wir konstruieren ein Problem, wo die Lösung garantiert positiv ist.
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
        
        # Test In-Place (Zero Allocation Check - später mit BenchmarkTools)
        ws = NNLSWorkspace(2, 2)
        status, x_inplace = nnls!(ws, A, b)
        
        @test status == :Success
        @test x_inplace ≈ x_exact rtol=1e-6
    end

    @testset "2. Boundary Solution (Active Constraints)" begin
        # Wir konstruieren ein Problem, wo die Unconstrained-Lösung negativ wäre.
        # Matrix A = I (Identität).
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
        @test x_nnls[2] ≈ 0.0 atol=1e-10 # Stellenwert an der Grenze
    end

    @testset "3. Rank Deficiency (Degenerierte Matrix)" begin
        # A ist singulär (Spalte 2 = 2 * Spalte 1)
        # b ist so gewählt, dass Lösung nicht eindeutig ist im Unconstrained Fall,
        # aber NNLS eine eindeutige Lösung finden muss (Mindestnorm oder Active Set).
        
        A = [1.0 2.0; 
             1.0 2.0] 
        b = [3.0, 3.0] # Passt perfekt zu x = [1, 1] oder x = [3, 0]
        
        # Wir testen hier vor allem, ob der Solver abstürzt (Rank Guard greift).
        # Ein gültiger NNLS-Solver sollte eine Lösung x >= 0 finden, die ||Ax-b|| minimiert.
        # Da A*[1, 1] = [3, 3], ist das Residuum 0.
        
        x_nnls = nnls(A, b)
        
        # Prüfen: Ist das Problem lösbar?
        residual = norm(A * x_nnls - b)
        @test residual < 1e-6
        @test all(x_nnls .>= -1e-10)
    end
    
    @testset "4. Zero Solution" begin
        # Fall, wo der optimale Punkt genau im Ursprung liegt.
        # b liegt vollständig außerhalb des positiven Kegels von A.
        
        A = [1.0 0.0; 
             0.0 1.0]
        b = [-1.0, -1.0]
        
        # Gradient bei 0 ist A' * (-b) = [-1, -1]. 
        # Warte, w = A'(b - Ax). Bei x=0 ist w = A'b = [-1, -1].
        # KKT: w muss <= 0 sein (Dual Feasibility).
        # Wenn w <= 0 ist, ist x=0 optimal.
        
        x_nnls = nnls(A, b)
        @test x_nnls ≈ [0.0, 0.0] atol=1e-10
    end

end
