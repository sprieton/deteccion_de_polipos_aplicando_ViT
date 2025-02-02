# Detección de Pólipos Aplicando Vision Transformers (ViT)

## Introducción

La detección temprana de pólipos en el tracto gastrointestinal es fundamental para prevenir el desarrollo de enfermedades graves como el cáncer colorrectal. En este contexto, los avances en la inteligencia artificial y, en particular, en las redes neuronales profundas, han permitido desarrollar sistemas de asistencia médica con un alto nivel de precisión. Este Trabajo de Fin de Grado se centra en la implementación y evaluación de un modelo de **Vision Transformers (ViT)** para la detección automática de pólipos en imágenes endoscópicas.

## Contexto y Motivación

Los Vision Transformers han emergido como una arquitectura innovadora en el campo de la visión por computadora. A diferencia de las redes convolucionales tradicionales (CNNs), los ViT aprovechan la autoatención para procesar imágenes como una secuencia de parches, lo que permite capturar relaciones globales entre regiones de una imagen. Esta capacidad es particularmente útil en la detección de pólipos, donde las características visuales pueden ser sutiles y dispersas.

El desarrollo de este proyecto surge de la necesidad de explorar métodos más efectivos y generalizables para la detección de anomalías en imágenes médicas, un área en la que los modelos basados en CNN han mostrado limitaciones en ciertos casos.

## Objetivos

1. **Implementar un modelo ViT** para la detección de pólipos en imágenes endoscópicas, ajustando los parámetros y arquitecturas según los requisitos del problema.
2. **Evaluar el rendimiento del modelo** utilizando métricas relevantes como precisión, sensibilidad, especificidad y F1-Score.
3. **Comparar el enfoque ViT** con métodos tradicionales basados en CNN para analizar las ventajas y desventajas de ambas aproximaciones.
4. **Generar un dataset anotado** y aplicar técnicas de preprocesamiento que optimicen el rendimiento del modelo.

## Contribuciones del Proyecto

Este TFG aporta al campo de la detección de anomalías en imágenes médicas mediante:
- La aplicación de una arquitectura moderna y poco explorada en este ámbito como los Vision Transformers.
- La creación de un pipeline reproducible que pueda ser empleado como base para futuros estudios en la detección automática de pólipos.
- Un análisis exhaustivo de los resultados obtenidos, proporcionando información valiosa sobre la viabilidad de ViT en aplicaciones médicas.
