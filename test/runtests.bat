echo Starte Projekt-Setup...
julia --project=. -e "using Pkg; Pkg.develop(PackageSpec(path=joinpath(@__DIR__,\"..\"))); Pkg.instantiate()"

echo Starte runtests...
julia --project=. runtests.jl

pause