@echo off
echo ================================================
echo  CoreNNLS.jl -- GitHub Push
echo ================================================
echo.

cd /d "%~dp0"

if not exist ".git" (
    echo Initialisiere Git-Repository...
    git init
    echo.
)

if not exist ".gitignore" (
    echo benchmark/problems.jld2 > .gitignore
    echo Manifest.toml >> .gitignore
    echo benchmark/Manifest.toml >> .gitignore
    echo test/Manifest.toml >> .gitignore
    echo *.jld2 >> .gitignore
    echo .gitignore angelegt.
    echo.
)

git remote get-url origin >nul 2>&1
if errorlevel 1 (
    set /p REMOTE_URL="GitHub-URL eingeben (z.B. https://github.com/DEINNAME/CoreNNLS.jl.git): "
    git remote add origin %REMOTE_URL%
)

git add .
git status
echo.

set /p COMMIT_MSG="Commit-Nachricht (Enter fuer 'Initial release v0.5.0'): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Initial release v0.5.0

git commit -m "%COMMIT_MSG%"
echo.

echo Pushe nach GitHub...
git push -u origin master

echo.
echo ================================================
echo  Fertig! Naechster Schritt: Registrierung
echo  Auf GitHub unter einem Commit kommentieren:
echo  @JuliaRegistrator register
echo ================================================
pause