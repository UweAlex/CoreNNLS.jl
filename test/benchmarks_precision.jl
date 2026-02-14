# test/benchmarks_precision.jl
using CoreNNLS, LinearAlgebra, Printf

# Hilfsfunktion zur Erzeugung von Problemstellungen mit bekannter Lösung (x_true = 1.0)
function generate_ill_conditioned_test(type::Symbol, n::Int, T::Type)
    A = zeros(T, n, n)
    if type == :Hilbert
        for i in 1:n, j in 1:n
            A[i,j] = inv(T(i + j - 1))
        end
    elseif type == :Vandermonde
        points = LinRange(T(0.1), T(0.9), n)
        for i in 1:n, j in 1:n
            A[i,j] = points[i]^(j-1)
        end
    elseif type == :Pascal
        # Pascal Matrix (Lower Triangular)
        for i in 1:n, j in 1:n
            if i >= j
                A[i,j] = binomial(i-1, j-1)
            end
        end
    end
    
    x_true = ones(T, n)
    b = A * x_true
    return A, b, x_true
end

function run_precision_benchmarks()
    # Konfiguration für den README-Output
    test_specs = [
        (:Pascal, 25, "Pascal (Lower)"),
        (:Hilbert, 12, "Hilbert Matrix"),
        (:Vandermonde, 15, "Vandermonde")
    ]
    
    results = []

    println("🚀 Running CoreNNLS Precision Benchmarks...")
    
    for (type, n, label) in test_specs
        # 1. Referenz-Daten in hoher Präzision erzeugen
        A_ref, b_ref, x_true = generate_ill_conditioned_test(type, n, BigFloat)
        c_val = cond(Float64.(A_ref))
        
        # 2. Lösung mit Float64
        # Wir nutzen die Standard-API von CoreNNLS
        x_f64 = nnls(Float64.(A_ref), Float64.(b_ref))
        err_f64 = norm(x_f64 - ones(n))
        
        # 3. Lösung mit BigFloat (256-bit für Geschwindigkeit im CI)
        setprecision(BigFloat, 256) do
            A_big = BigFloat.(A_ref)
            b_big = BigFloat.(b_ref)
            
            # Workspace-Ansatz (Forensic Layer)
            ws = NNLSWorkspace(n, n, BigFloat)
            status, x_big = nnls!(ws, A_big, b_big)
            
            err_big = norm(x_big - ones(BigFloat, n))
            push!(results, (label, n, c_val, err_f64, Float64(err_big)))
        end
    end

    # --- GENERATE README MARKDOWN OUTPUT ---
    println("\n" * "─"^60)
    println("### COPY THE FOLLOWING INTO YOUR README.md")
    println("─"^60 * "\n")
    
    @printf "| Matrix Type | n | Cond(A) | Float64 Error | BigFloat Error | Gain |\n"
    @printf "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for (label, n, c, e64, ebig) in results
        gain = e64 / (ebig + eps(BigFloat))
        @printf "| %s | %d | %.1e | %.2e | %.2e | **10^{%.1f}** |\n" label n c e64 ebig log10(gain)
    end
    
    println("\n" * "─"^60)
end

# Führt den Benchmark aus
run_precision_benchmarks()
