
echo Starte Projekt-Setup...
julia --project=. -e "using Pkg; Pkg.develop(PackageSpec(path=joinpath(@__DIR__,\"..\"))); Pkg.instantiate()"

echo Starte test_householder.jl...
julia --project=. test_householder.jl

pause