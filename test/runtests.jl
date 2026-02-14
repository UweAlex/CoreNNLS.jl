using Test
using LinearAlgebra
using CoreNNLS
using Random

@testset "CoreNNLS.jl - Complete Forensic Suite" begin
    
    # ==========================================================================
    # SECTION 1: BASIC FUNCTIONALITY & INTERIOR
    # ==========================================================================
    @testset "1. Interior & Boundary Basics" begin
        # Erwartet: x = [16/7, 10/7]
        @test nnls([1.0 0.5; 0.5 2.0], [3.0, 4.0]) ≈ [16/7, 10/7]
        
        # Test auf strikte Projektion (x2 muss 0 sein)
        x_boundary = nnls([1.0 0.0; 0.0 1.0], [1.0, -1.0])
        @test x_boundary ≈ [1.0, 0.0] atol=1e-12
        
        # Negatives Ziel -> Resultat [0, 0]
        @test nnls(diagm(ones(2)), [-1.0, -2.0]) ≈ [0.0, 0.0] atol=1e-12
    end

    # ==========================================================================
    # SECTION 2: REFERENCE EQUIVALENCE (SCIPY / NETLIB)
    # ==========================================================================
    @testset "2. Reference: SciPy/Netlib" begin
        # TestCase 1: Negatives Ziel
        A_sp = [1.0 2.0; 3.0 4.0; 5.0 6.0]; b_sp = [-1.0, -2.0, -3.0]
        @test nnls(A_sp, b_sp) ≈ [0.0, 0.0] atol=1e-15

        # TestCase 2: Standard Interior (x1=x2=5/3)
        A_sp2 = [1.0 0.0; 1.0 1.0; 0.0 1.0]; b_sp2 = [2.0, 3.0, 2.0]
        @test nnls(A_sp2, b_sp2) ≈ [5/3, 5/3] atol=1e-12
    end

    # ==========================================================================
    # SECTION 3: SCALING STABILITY
    # ==========================================================================
    @testset "3. Reference: Scaling Stability" begin
        A = [1e3 0.0; 0.0 1.0]; b = [1e3, 1.0]
        @test nnls(A, b) ≈ [1.0, 1.0] rtol=1e-5
        
        A_ext = [1e10 0.0; 0.0 1e-10]; b_ext = [1e10, 1e-10]
        x_ext = nnls(A_ext, b_ext)
        @test all(isfinite, x_ext)
        @test all(x_ext .>= -1e-9)
    end

    # ==========================================================================
    # SECTION 4: EDGE CASES & RANK DEFICIENCY
    # ==========================================================================
    @testset "4. Zero Column (Null Column)" begin
        A = [1.0 0.0; 1.0 0.0; 0.0 0.0]; b = [1.0, 2.0, 3.0]
        ws = NNLSWorkspace(3, 2)
        status, x = nnls!(ws, A, b)
        @test status == :Success
        # Korrigierte Syntax für das Makro
        @test x ≈ [1.5, 0.0] atol=1e-16 
    end

    @testset "5. Collinearity & Rank Deficiency" begin
        A = [1.0 1.0; 1.0 1.0; 1.0 1.0]; b = [2.0, 2.0, 2.0]
        ws = NNLSWorkspace(3, 2)
        status, x = nnls!(ws, A, b)
        @test status == :Success
        @test x[1] + x[2] ≈ 2.0 
        @test all(x .>= -1e-12)
    end

    # ==========================================================================
    # SECTION 7: HIGH PRECISION
    # ==========================================================================
    @testset "7. High-Precision (BigFloat)" begin
        T = BigFloat
        setprecision(BigFloat, 256) do
            n = 4 # Hilbert ist bei n=8 schon extrem instabil
            A = T[inv(T(i + j - 1)) for i in 1:n, j in 1:n]
            x_true = ones(T, n); b = A * x_true
            ws = NNLSWorkspace(n, n, T; w_tol = T(1e-60), rank_tol = T(1e-60))
            status, x = nnls!(ws, A, b)
            @test status == :Success
            @test norm(x - x_true) < 1e-20 
        end
    end

    # ==========================================================================
    # SECTION 10: MASS STRESS TEST (HARD KKT VALIDATION)
    # ==========================================================================
    @testset "10. Mass Stress Test (KKT Validator)" begin
        Random.seed!(42)
        for _ in 1:100
            m, n = rand(10:50), rand(5:20)
            A = randn(m, n); b = randn(m)
            x = nnls(A, b)
            
            # Die mathematische Wahrheit prüfen:
            # 1. Primal Feasibility
            @test all(x .>= -1e-11)
            
            # 2. Dual Feasibility & Complementary Slackness
            # w = A' * (b - A*x)
            w = A' * (b - A*x)
            for j in 1:n
                if x[j] > 1e-8
                    @test abs(w[j]) < 1e-5  # Falls x > 0, muss w = 0 sein
                else
                    @test w[j] < 1e-5       # Falls x = 0, darf w nicht positiv ziehen
                end
            end
        end
    end

    # ==========================================================================
    # SECTION 13: BURKARDT/LAWSON (RECOVERY & OPTIMALITY)
    # ==========================================================================
    @testset "13. Burkardt/Lawson Fortran Recovery" begin
        Random.seed!(42)
        for anoise in [0.0, 1e-4]
            for m in 3:6, n in 2:4
                A = randn(m, n)
                x_seed = abs.(randn(n))
                b = A * x_seed + anoise * randn(m)
                x = nnls(A, b)
                
                # Immer prüfen: Ist die Lösung optimal?
                w = A' * (b - A * x)
                for j in 1:n
                    if x[j] > 1e-7
                        @test abs(w[j]) < 1e-4 
                    else
                        @test w[j] < 1e-4      
                    end
                end

                # Identität nur prüfen, wenn das System eindeutig ist (m >= n)
                if m >= n && anoise == 0.0 && cond(A) < 1e8
                    @test x ≈ x_seed atol=1e-10
                end
            end
        end
    end

end
