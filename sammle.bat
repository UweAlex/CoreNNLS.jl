@echo off
setlocal EnableDelayedExpansion

set "outputFile=alles.txt"

echo Leere oder erstelle %outputFile%...
> "%outputFile%" echo.

echo Sammle Inhalte von *.jl *.bat *.toml Dateien rekursiv...

for /r . %%F in (*.jl *.bat *.toml) do (
    REM Überspringe die Ausgabedatei selbst und das aktuelle Skript
    if /I "%%~nxF" neq "%~nx0" if /I "%%~nxF" neq "%outputFile%" (
        echo Pfad:       %%F >> "%outputFile%"
        echo Dateiname:  %%~nxF >> "%outputFile%"
        echo. >> "%outputFile%"
        echo Inhalt: >> "%outputFile%"
        type "%%F" >> "%outputFile%" 2>nul
        echo. >> "%outputFile%"
        echo --- Ende von %%~nxF --- >> "%outputFile%"
        echo. >> "%outputFile%"
        echo. >> "%outputFile%"
    )
)

if %errorlevel% neq 0 (
    echo Fehler beim Sammeln der Dateien!
    echo (eventuell sind gar keine .jl / .bat / .toml Dateien vorhanden)
) else (
    echo.
    echo Fertig - alles in %outputFile% gespeichert.
)

pause
endlocal