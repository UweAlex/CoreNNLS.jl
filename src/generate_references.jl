"""
generate_references.jl

45 Referenzlösungen (5 Klassen × 3×3 m/n-Raster) in 1024-bit BigFloat.

m ∈ {30, 120, 400}   (Zeilen)
n ∈ {8,   30,  100}  (Spalten)

Einmalig ausführen:  julia --project=. generate_references.jl
"""

using Pkg; Pkg.add("JLD2"; io=devnull)
using LinearAlgebra, Random, JLD2, CoreNNLS

const M_VALS = [30, 120, 400]
const N_VALS = [8,   30, 100]

# =============================================================================
# Matrixgeneratoren
# =============================================================================

function make_random(T,m,n,seed)
    rng=MersenneTwister(seed); T.(randn(rng,m,n)), T.(randn(rng,m))
end
function make_sparse(T,m,n,seed; density=0.4)
    rng=MersenneTwister(seed)
    A=zeros(T,m,n)
    for j in 1:n, i in 1:m; rand(rng)<density && (A[i,j]=T(randn(rng))); end
    A, T.(randn(rng,m))
end
function make_vandermonde(T,m,n)
    nodes=range(T("0.1"),T("0.9"),length=m)
    [nodes[i]^(j-1) for i in 1:m, j in 1:n], ones(T,m)
end
function make_hilbert(T,m,n)
    k=min(m,n); [T(1)/(i+j-1) for i in 1:k, j in 1:k], ones(T,k)
end
function make_pascal(T,m,n)
    k=min(min(m,n),12)
    T.([binomial(i+j-2,i-1) for i in 1:k, j in 1:k]), ones(T,k)
end

function make_bf(cls,m,n,seed)
    T=BigFloat
    cls==:random      && return make_random(T,m,n,seed)
    cls==:sparse      && return make_sparse(T,m,n,seed)
    cls==:vandermonde && return make_vandermonde(T,m,n)
    cls==:hilbert     && return make_hilbert(T,m,n)
    cls==:pascal      && return make_pascal(T,m,n)
end

prob_name(cls,m,n) = begin
    k=min(m,n); kp=min(k,12)
    cls==:hilbert ? "Hilbert_$(k)x$(k)" :
    cls==:pascal  ? "Pascal_$(kp)x$(kp)" :
    "$(uppercasefirst(string(cls)))_$(m)x$(n)"
end

# =============================================================================
# Referenzen berechnen
# =============================================================================

println("Berechne 45 Referenzlösungen (1024-bit BigFloat)...")
println("=" ^ 60)

references = Dict{String,Vector{Float64}}()
meta       = Dict{String,Any}()
setprecision(1024) do
    idx = 0
    for cls in [:random,:sparse,:vandermonde,:hilbert,:pascal]
        for m in M_VALS, n in N_VALS
            idx += 1
            name = prob_name(cls,m,n)
            print("[$idx/45] $name ... "); flush(stdout)
            seed = idx * 97 + Int(hash(cls) % 1000)
            A,b  = make_bf(cls,m,n,seed)
            t    = @elapsed x = nnls(A,b)
            res  = Float64(norm(b - A*x))
            references[name] = Float64.(x)
            meta[name] = (cls=string(cls), m=size(A,1), n=size(A,2),
                          res_bigfloat=res, seed=seed)
            println("OK  ‖r‖=$(round(res,sigdigits=3))  $(round(t*1000,digits=1))ms")
        end
    end
end

jldsave(joinpath(@__DIR__,"references.jld2"); references=references, meta=meta,
        m_vals=M_VALS, n_vals=N_VALS)
println("\n45 Referenzlösungen → references.jld2")