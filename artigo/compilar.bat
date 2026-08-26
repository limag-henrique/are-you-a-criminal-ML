@echo off
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 goto erro
bibtex main
if errorlevel 1 goto erro
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 goto erro
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 goto erro
echo.
echo Compilacao concluida: main.pdf
pause
exit /b 0
:erro
echo.
echo Ocorreu um erro na compilacao. Consulte main.log.
pause
exit /b 1
