# Pronostico de alcance de Instagram con Series temporales y SARIMA

Este proyecto utiliza técnicas de Series temporales prever el alcance de instagram en los proximos meses en base a los datos historicos ya existentes. Además, se incluye un análisis de las estadisticas del alcande de Instagra como la media, mediana y desviacion estandar por cada dia de la semana y demas tendencias y distribuciones.

## Objetivo

El objetivo principal de este proyecto es implementar algoritmos Series temporales para:

- Prever las tendencias futuras del alcance.
- Pronosticar y aprovechar oportunidades de mejorar e identificar tendencias mas exitosas.

## Funcionalidades

### 1. Pronostico de tendencias

El script `Modelo SARIMA.py` realiza en primer lugar el hallazgo de los valores 'p, d y q' para implementar en el modelo, luego con esos valores se realiza el modelo de Series temporales SARIMA y se grafica las tendencias futuras.

### 2. Análisis previos y estadisticas

El archivo `Estadisticas y analisis previos.ipynb` incluye análisis preliminares de los datos y confeccion de estadisticas para interpretar patrones de tendencia claves en base a los datos historicos.

### 3. Archivo fuente

El archivo `Instagram-Reach.csv` provee los datos tabulares para desarrollar el analisis y modelo posterior.

## Estructura del Proyecto:

- Analisis preliminar

- Correlacion estadisticas

- Hallazgo de los valores p,d y q

- Modelo de pronostico de tendencias SARIMA
