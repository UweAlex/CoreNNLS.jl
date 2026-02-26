echo Starte Projekt-Setup...
julia --project=. -e "using Pkg; Pkg.develop(PackageSpec(path=joinpath(@__DIR__,\"..\"))); Pkg.instantiate(); Pkg.add([\"BenchmarkTools\", \"NonNegLeastSquares\"])"

echo Starte generate_problems.jl...
julia --project=. generate_problems.jl

pause