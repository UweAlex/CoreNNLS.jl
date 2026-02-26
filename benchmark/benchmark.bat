
echo Starte Projekt-Setup...
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.add([\"BenchmarkTools\", \"NonNegLeastSquares\"])"

echo Starte Benchmark...
julia --project=. Benchmark.jl

pause