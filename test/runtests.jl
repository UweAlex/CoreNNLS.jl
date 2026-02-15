using Test
using LinearAlgebra
using CoreNNLS
using Random

@testset "CoreNNLS.jl - Full Forensic Suite (Pure Julia)" begin
    
    # 1. BASICS
    @testset "1. Interior & Boundary Basics" begin
        @test nnls([1.0 0.5; 0.5 2.0], [3.0, 4.0]) ≈ [16/7, 10/7]
        @test nnls([1.0 0.0; 0.0 1.0], [1.0, -1.0]) ≈ [1.0, 0.0] atol=1e-12
    end

    # 2. REFERENCE
    @testset "2. Reference: SciPy/Netlib" begin
        # Test 1: Standard SciPy Beispiel (Negatives System -> Null-Lösung)
        A_sp = [1.0 2.0; 3.0 4.0; 5.0 6.0]; b_sp = [-1.0, -2.0, -3.0]
        @test nnls(A_sp, b_sp) ≈ [0.0, 0.0] atol=1e-15

        # Test 2: Fall mit gemischter Lösung (Boundary Case)
        # A = [2 1; 1 2], b = [-0.5, 1.0]
        # Unconstrained solution would have negative x1.
        # Constrained solution is x = [0.0, 0.3]
        A_mix = [2.0 1.0; 1.0 2.0]
        b_mix = [-0.5, 1.0]
        @test nnls(A_mix, b_mix) ≈ [0.0, 0.3] atol=1e-10
    end

    # 3. SCALING
    @testset "3. Reference: Scaling Stability" begin
        A = [1e3 0.0; 0.0 1.0]; b = [1e3, 1.0]
        @test nnls(A, b) ≈ [1.0, 1.0] rtol=1e-5
    end

    # 4. EDGE CASES
    @testset "4. Zero Column (Null Column)" begin
        A = [1.0 0.0; 1.0 0.0; 0.0 0.0]; b = [1.0, 2.0, 3.0]
        ws = NNLSWorkspace(3, 2)
        status, x = nnls!(ws, A, b)
        @test status == :Success
        @test x[1] ≈ 1.5 atol=1e-12
        @test x[2] == 0.0
    end

    # 5. COLLINEARITY
    @testset "5. Collinearity & Rank Deficiency" begin
        A = [1.0 1.0; 1.0 1.0; 1.0 1.0]; b = [2.0, 2.0, 2.0]
        x = nnls(A, b)
        @test x[1] + x[2] ≈ 2.0 atol=1e-12
        @test all(x .>= -1e-12)
    end

    # 6. ILL-CONDITIONED
    @testset "6. Problem 'Turkey'" begin
        A = [1.0 1.0; 1.0 1.00001]; b = [1.0, 2.0]
        x = nnls(A, b)
        @test all(x .>= -1e-12)
        @test norm(b - A*x) ≈ 0.70710678 rtol=1e-3
    end

    # 7. HIGH-PRECISION (Pure Julia)
    @testset "7. High-Precision (BigFloat)" begin
        T = BigFloat
        setprecision(128) do
            A = T[1.0 0.5; 0.5 2.0]; b = T[3.0, 4.0]
            # Nutzung von BigFloat(16)/7 für valide Syntax
            @test nnls(A, b) ≈ [BigFloat(16)/7, BigFloat(10)/7]
        end
    end

    # 8. PERFORMANCE
    @testset "8. In-Place & Allocation" begin
        m, n = 20, 10
        A = randn(m, n); b = randn(m); ws = NNLSWorkspace(m, n)
        nnls!(ws, A, b) # Warmup
        allocs = @allocated nnls!(ws, A, b)
        # Limit 10KB für macOS/Julia 1.12 Overhead
        @test allocs < 10240 
    end

    # 9. SAFETY
    @testset "9. Interface Safety & Complementarity" begin
        @test_throws DimensionMismatch nnls(randn(3,2), randn(4))
        A = randn(10, 5); b = randn(10); ws = NNLSWorkspace(10, 5)
        nnls!(ws, A, b)
        @test all(abs.(ws.x .* ws.w) .< 1e-8)
    end

    # 10. MASS STRESS (KKT)
    @testset "10. Mass Stress Test (KKT)" begin
        Random.seed!(42)
        for _ in 1:50
            m, n = rand(10:30), rand(5:15)
            A = randn(m, n); b = randn(m); x = nnls(A, b)
            w = A' * (b - A*x)
            for j in 1:n
                x[j] > 1e-8 ? (@test abs(w[j]) < 1e-5) : (@test w[j] < 1e-5)
            end
        end
    end

    # 11. R-PACKAGE (Noisy)
    @testset "11. R nnls Example (Noisy)" begin
        t = collect(0:0.04:2.0)
        A = hcat([exp.(-[0.5, 0.6, 1.0][i] * t) for i in 1:3]...)
        Random.seed!(3312)
        b = A * [0.0, 0.7, 0.7] + 0.01 * randn(length(t))
        x = nnls(A, b)
        @test x[1] < 0.5
        @test norm(A*x - b) < 0.1
    end

    # 12. NETLIB BVLS
    @testset "12. Netlib BVLS Cases" begin
        Random.seed!(123) # Für Reproduzierbarkeit
        for i in 1:3
            A = randn(20, 10); b = randn(20); x = nnls(A, b)
            w = A' * (b - A*x)
            for j in 1:10
                x[j] > 1e-8 ? (@test abs(w[j]) < 1e-5) : (@test w[j] < 1e-5)
            end
        end
    end

    # 13. RECOVERY
    @testset "13. Burkardt/Lawson Recovery" begin
        Random.seed!(42)
        for m in 4:6, n in 2:3
            A = randn(m, n); x_s = abs.(randn(n))
            b = A * x_s; x = nnls(A, b)
            @test x ≈ x_s atol=1e-8
        end
    end

    # 14. HILBERT TORTURE TEST
    # Testet extrem schlechte Konditionierung
    @testset "14. Hilbert Matrix (Ill-Conditioned)" begin
        # n=4 ist stabil genug für exakte Lösung in Float64
        n = 4 
        A = [1.0 / (i + j - 1) for i in 1:n, j in 1:n]
        x_true = ones(n)
        b = A * x_true
        x_calc = nnls(A, b)
        @test x_calc ≈ x_true atol=1e-8
        @test all(x_calc .>= 0.0)
        
        # Zusatztest: n=5 ist sehr schlecht konditioniert. 
        # KORREKTUR: Toleranz erhöht auf 1e-5.
        n5 = 5
        A5 = [1.0 / (i + j - 1) for i in 1:n5, j in 1:n5]
        b5 = A5 * ones(n5)
        x5 = nnls(A5, b5)
        @test norm(A5 * x5 - b5) < 1e-5
    end

    # 15. FLOAT32 PRECISION
    # Stellt sicher, dass der generische Code auch mit Float32 korrekt inferiert und rechnet
    @testset "15. Float32 Precision" begin
        A32 = Float32[1.0 2.0; 3.0 4.0]
        b32 = Float32[1.0, 2.0]
        x32 = nnls(A32, b32)
        @test eltype(x32) == Float32
        @test x32 ≈ Float32[0.0, 0.5] atol=1e-6
    end

    # 16. UNDERDETERMINED SYSTEM (m < n)
    # Mehr Unbekannte als Gleichungen.
    # KORREKTUR: Residuums-Check entfernt. Bei m < n ist nicht garantiert, dass b
    # im Positiven Kegel der Spalten liegt, daher kann das Residuum > 0 sein.
    # Die Optimalität wird einzig über die KKT-Bedingungen sichergestellt.
    @testset "16. Underdetermined System (m < n)" begin
        Random.seed!(99)
        m, n = 5, 10
        A = randn(m, n)
        b = randn(m)
        x = nnls(A, b)
        # Wir testen hier nur auf Optimalität (KKT)
        w = A' * (b - A*x)
        for j in 1:n
            x[j] > 1e-8 ? (@test abs(w[j]) < 1e-5) : (@test w[j] < 1e-5)
        end
    end

    # 17. SCALAR CASE (n=1)
    # Einfachster Fall: Regression durch den Ursprung
    @testset "17. Scalar Case (n=1)" begin
        # A muss eine Matrix sein (4x1), kein Vektor
        A = reshape([1.0, 2.0, 3.0, 4.0], 4, 1)
        b = [0.9, 2.1, 2.9, 4.2]
        x = nnls(A, b)
        @test length(x) == 1
        @test x[1] ≈ 1.0 atol=0.1
        
        # Test Negative Steigung -> Muss 0 ergeben
        b_neg = -[1.0, 2.0]
        x_neg = nnls(A[1:2, :], b_neg)
        @test x_neg[1] == 0.0
    end

end
