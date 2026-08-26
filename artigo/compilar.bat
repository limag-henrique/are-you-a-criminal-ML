@echo off
where tectonic >nul 2>nul
if %errorlevel% equ 0 (
    tectonic -X compile main.tex
    goto fim
)
if exist "%LOCALAPPDATA%\Programs\tectonic\tectonic.exe" (
    "%LOCALAPPDATA%\Programs\tectonic\tectonic.exe" -X compile main.tex
    goto fim
)

pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 goto erro
bibtex main
if errorlevel 1 goto erro
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 goto erro
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 goto erro

:fim
echo.
echo Compilacao concluida: main.pdf
pause
exit /b 0

:erro
echo.
echo Ocorreu um erro na compilacao. Consulte main.log.
pause
exit /b 1
