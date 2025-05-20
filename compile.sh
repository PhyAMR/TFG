#!/bin/bash

# Ejecuta pdflatex para compilar el documento LaTeX
pdflatex main.tex

# Ejecuta bibtex para procesar las referencias bibliográficas
bibtex main

# Ejecuta pdflatex dos veces más para actualizar las referencias
pdflatex main.tex
pdflatex main.tex

echo "Compilación completa: main.pdf generado."
