@echo off
echo ================================================
echo  CoreNNLS.jl -- GitHub Push
echo ================================================
echo.

REM Ins Repo-Root wechseln (ein Verzeichnis über dieser .bat-Datei)
cd /d "%~dp0"

REM Prüfen ob git initialisiert ist
if not exist ".git" (
    echo Initialisiere Git-Repository...
    git init
    echo.
)

REM .gitignore anlegen falls nicht vorhanden
if not exist ".gitignore" (
    echo benchmark/problems.jld2 > .gitignore
    echo Manifest.toml >> .gitignore
    echo benchmark/Manifest.toml >> .gitignore
    echo test/Manifest.toml >> .gitignore
    echo *.jld2 >> .gitignore
    echo .gitignore angelegt.
    echo.
)

REM Remote setzen -- bitte URL anpassen falls nötig
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    set /p REMOTE_URL="GitHub-URL eingeben (z.B. https://github.com/DEINNAME/CoreNNLS.jl.git): "
    git remote add origin %REMOTE_URL%
)

REM Alles stagen
git add .
git status
echo.

REM Commit
set /p COMMIT_MSG="Commit-Nachricht (Enter fuer 'Initial release v0.5.0'): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Initial release v0.5.0

git commit -m "%COMMIT_MSG%"
echo.

REM Push
echo Pushe nach GitHub...
git push -u origin main
if errorlevel 1 (
    echo.
    echo Tipp: Falls 'main' nicht existiert, versuche:
    echo   git push -u origin master
)

echo.
echo ================================================
echo  Fertig! Naechster Schritt: Registrierung
echo  Auf GitHub unter einem Commit kommentieren:
echo  @JuliaRegistrator register
echo ================================================
pause