# test/test_householder.jl
#
# Testet reflector!, reflector_apply!, householder_qr!, householder_apply_qt!, back_substitute!
# nach dem Vorbild von Julia stdlib LinearAlgebra/test/
#
# Ausführen:  julia --project=. test/test_householder.jl

using Test
using LinearAlgebra

# Modulpfad anpassen damit include("src/householder.jl") funktioniert
include(joinpath(@__DIR__, "..", "src", "householder.jl"))

@testset "Householder Unterroutinen" begin

    # =========================================================================
    @testset "reflector! — Grundeigenschaft: H*x = β*e₁" begin
        for T in (Float64, Float32, BigFloat)
            for x_orig in [
                T[3.0, 4.0],                        # einfach, positives x[1]
                T[-3.0, 4.0],                        # negatives x[1]
                T[1.0, 0.0, 0.0],                    # bereits e₁
                T[0.0, 1.0],                         # x[1] = 0
                T[1e-8, 1e-8],                       # klein
                T[1e8,  1e8],                        # groß
                T[1.0, 2.0, 3.0, 4.0],              # n=4
            ]
                x = copy(x_orig)
                τ = reflector!(x)
                β = x[1]                             # nach dem Aufruf ist x[1] = β

                # Reflektorvektor v = [1; x[2:end]]
                v = similar(x_orig)
                v[1] = one(T)
                v[2:end] .= x[2:end]

                # H = I - τ*v*v'
                n = length(x_orig)
                H = Matrix{T}(I, n, n) - τ * v * v'

                # Eigenschaft 1: H ist orthogonal  (H'H = I)
                @test H'*H ≈ Matrix{T}(I, n, n)  atol=100*eps(T)

                # Eigenschaft 2: H*x_orig = β*e₁
                # Toleranz skaliert mit Norm von x (wichtig für große Werte wie 1e8)
                result = H * x_orig
                e1 = zeros(T, n); e1[1] = β
                scale = max(norm(x_orig), one(T))
                @test result ≈ e1  atol=100*eps(T)*scale

                # Eigenschaft 3: τ ≥ 0 (Vorzeichen-Konvention)
                @test τ >= zero(T)
            end
        end
    end

    # =========================================================================
    @testset "reflector! — Nullvektor gibt τ=0 zurück" begin
        for T in (Float64, Float32, BigFloat)
            x = zeros(T, 3)
            τ = reflector!(x)
            @test τ == zero(T)
        end
    end

    # =========================================================================
    @testset "reflector! — Skalar (n=1)" begin
        for T in (Float64, Float32, BigFloat)
            x = T[5.0]
            τ = reflector!(x)
            # Einzel-Element: Reflektor ist Identität oder Spiegelung
            # H*x_orig = x[1]*e₁ — trivial erfüllt
            @test τ >= zero(T)
        end
    end

    # =========================================================================
    @testset "reflector_apply! — Anwendung auf Matrix" begin
        for T in (Float64, Float32, BigFloat)
            # Baue manuell einen Reflektor und wende ihn an
            x_orig = T[3.0, 4.0, 0.0]
            x = copy(x_orig)
            τ = reflector!(x)
            β = x[1]
            v = [one(T); x[2:end]]

            A = T[1.0 2.0 3.0;
                  4.0 5.0 6.0;
                  7.0 8.0 9.0]
            A_copy = copy(A)

            # Manuell: H*A via dichter Matrix
            H = Matrix{T}(I, 3, 3) - τ * v * v'
            expected = H * A_copy

            # Unsere Funktion: A ← H*A
            reflector_apply!(x, τ, A)

            @test A ≈ expected  atol=100*eps(T)
        end
    end

    # =========================================================================
    @testset "householder_qr! — QR-Zerlegung A = Q*R" begin
        for T in (Float64, Float32, BigFloat)
            for (m, n_p) in [(4,2), (3,3), (5,3), (2,2)]
                # Zufällige (aber reproduzierbare) Matrix
                A_orig = T.(reshape(collect(1.0:m*n_p), m, n_p) .+ T(0.1) .* T.(collect(1:m) * collect(1:n_p)'))
                tau    = zeros(T, n_p)
                A      = copy(A_orig)

                householder_qr!(A, tau, m, n_p)

                # R liegt im oberen Dreieck von A[1:n_p, 1:n_p]
                R = UpperTriangular(A[1:n_p, 1:n_p])

                # Q rekonstruieren: Q = H_1 * H_2 * ... * H_{n_p}
                Q = Matrix{T}(I, m, m)
                for k in n_p:-1:1
                    v = zeros(T, m - k + 1)
                    v[1] = one(T)
                    v[2:end] .= A[(k+1):m, k]
                    Hk = Matrix{T}(I, m-k+1, m-k+1) - tau[k] * v * v'
                    # Einbetten in m×m
                    Hk_full = Matrix{T}(I, m, m)
                    Hk_full[k:m, k:m] = Hk
                    Q = Hk_full * Q
                end

                # A_orig = Q * [R; 0]
                QR_reconstructed = Q * [Matrix(R); zeros(T, m-n_p, n_p)]
                @test A_orig ≈ QR_reconstructed  atol=1000*eps(T)

                # Q ist orthogonal
                @test Q'*Q ≈ Matrix{T}(I, m, m)  atol=1000*eps(T)
            end
        end
    end

    # =========================================================================
    @testset "householder_apply_qt! — Q'b korrekt" begin
        for T in (Float64, Float32, BigFloat)
            m, n_p = 4, 2
            A_orig = T[2.0 1.0; 1.0 3.0; 0.0 1.0; 1.0 0.0]
            b      = T[1.0, 2.0, 3.0, 4.0]
            tau    = zeros(T, n_p)
            A      = copy(A_orig)

            householder_qr!(A, tau, m, n_p)

            # Q rekonstruieren (wie oben)
            Q = Matrix{T}(I, m, m)
            for k in n_p:-1:1
                v = zeros(T, m - k + 1)
                v[1] = one(T)
                v[2:end] .= A[(k+1):m, k]
                Hk = Matrix{T}(I, m-k+1, m-k+1) - tau[k] * v * v'
                Hk_full = Matrix{T}(I, m, m)
                Hk_full[k:m, k:m] = Hk
                Q = Hk_full * Q
            end

            expected_qtb = Q' * b

            b_copy = copy(b)
            householder_apply_qt!(A, tau, b_copy, m, n_p)

            @test b_copy ≈ expected_qtb  atol=1000*eps(T)
        end
    end

    # =========================================================================
    @testset "back_substitute! — löst R*x = b" begin
        for T in (Float64, Float32, BigFloat)
            R = T[2.0 1.0; 0.0 3.0]
            b = T[5.0, 6.0]
            x = copy(b)
            back_substitute!(R, x, 2)
            # Lösung: x[2] = 2.0, x[1] = (5 - 1*2)/2 = 1.5
            @test x ≈ T[1.5, 2.0]  atol=100*eps(T)

            # Probe: R*x ≈ b_orig
            @test R * x ≈ b  atol=100*eps(T)
        end
    end

    # =========================================================================
    @testset "Vollständige Pipeline: QR löst Ax=b (Least Squares)" begin
        # A = [1 0.5; 0.5 2], b = [3, 4] → x = [16/7, 10/7]
        for T in (Float64, Float32, BigFloat)
            A_orig = T[1.0 0.5; 0.5 2.0]
            b_orig = T[3.0, 4.0]
            m, n   = 2, 2
            tau    = zeros(T, n)
            A      = copy(A_orig)

            householder_qr!(A, tau, m, n)

            b = copy(b_orig)
            householder_apply_qt!(A, tau, b, m, n)

            x = copy(b[1:n])
            back_substitute!(A, x, n)

            expected = T[16, 10] ./ T(7)
            @test x ≈ expected  rtol=1000*eps(T)
        end
    end

end  # @testset "Householder Unterroutinen"

println("\nAlle Householder-Tests bestanden!")