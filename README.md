# Evaluación de la Robustez y Efectividad de Modelos Fundacionales para la Detección de Pólipos
**Trabajo Fin de Grado** - _Grado en Ingeniería de Robótica Software (URJC)_

Autor: _Santiago Prieto Núñez_

Curso: _2024/2025_

## 📋 Contexto y motivación 

El cáncer colorrectal representa aproximadamente el **10% de los casos de cáncer a nivel mundial** y es la segunda causa de muerte por cáncer cada año. No obstante, puede prevenirse en gran medida mediante un diagnóstico temprano y un tratamiento adecuado. En este contexto, **el uso de herramientas de Inteligencia Artificial (IA)** para la detección precoz de pólipos durante las colonoscopias **se ha consolidado como una estrategia eficaz**, mejorando la identificación de
lesiones y apoyando la toma de decisiones clínicas.


## 🔨 Descripción del contenido 

El software disponible en **este repositorio contiene las herramientas para entrenar y analizar la robustez y efectividad de diversos modelos** fundacionales basados en Vision Transformers (ViT) y convolutional neural networks (CNN) para la detección automática de pólipos en imágenes de colonoscopias, **Además de herramientas para el análisis y procesado de datasets de imágen médica** en este contexto como es especialmente el caso del dataset de [Piccolo](https://www.biobancovasco.bioef.eus/en/Sample-and-data-e-catalog/Databases/PD178-PICCOLO-EN1.html).


## 📄 Licencia 

Este proyecto está licenciado bajo la [Licencia Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

Esto implica que **el código incluido se puede utilizar, modificar y distribuir, incluso con fines comerciales**, siempre que:
- Incluyas un aviso de copyright y la licencia original.
- Indiques si realizaste cambios.
- No utilices las marcas registradas del autor sin permiso.
- No responsabilices al autor por posibles daños derivados del uso del software.

Consulta el archivo [`LICENSE`](./LICENSE) para más detalles.

## 🧠 Uso del proyecto 

El proyecto completo incluye ejemplos de como implementar un entrenamiento completo de forma sencilla de los siguientes modelos:
- _YOLOv8_
- _Densenet121_
- _Yolos-base_
- _ViT-Small_
- _ViT-Large_
- _DeiT_
- _PVT_

A distintos conjuntos de datos provenientes de distintos dataset o solo uno y además analizar sus resultados con gráficas fácilmente. La intención del autor es que revisando estos ejemplos y junto a la explicación general de las dos principales herramientas proporcionadas,  se pueden analizar fácilmente nuevas arquitectras o utilizar el código como base para proyectos mayores o con otros datos.

### 👀 Image Dataset Processor

El primer **desafío a la hora de trabajar con imágenes** de colonoscopias es **la diversidad de los datos y fuentes**, lo que naturalmente resulta en una diversidad tanto de formatos como organización y en general naturaleza de estas imágenes. Por lo que en este entorno tan variable es necesario unificar los datos en un mismo conjunto. **Para procesar estos de forma eficiente, estandarizar su formato**, analizar su composición y así poder utilizar más o menos datos según sea necesario haciendo una interfaz sencilla de utilizar, **hemos desarrollado la clase Image Dataset Processor (IDP)**.

Esta clase procesa el dataset dado guardándolo como un diccionario de elementos. Ofrece herramientas y funciones para a partir de este diccionario salvar, modificar o obtener información relevante del mismo.

**A continaución explico un ejemplo** de su uso para el caso de cargar el dataset
de Piccolo siguiendo esta serie de pasos:

#### 1️⃣ **Inicializar la clase** 
  Empezando con los parámetros que deseamos, nombre o resolución como vemos en la imagen. también puedes hacer uso de parámetros como: **json_path para dar un nombre personalizado al JSON donde guardar los datos** del dataset una vez procesado, metadata_path con metadatos clínicos, en el caso de piccolo para poder dar un análisis más exhaustivo de la composición del dataset.
<p align="center">
  <img src="https://github.com/user-attachments/assets/e8d54289-7435-45e7-8c87-22306dd0a9fe" width="700" alt="IDP_iniciar_clase"/>
</p>

#### 2️⃣ **Cargar las imágenes en la clase**
  Para ello sencillamente debemos **introducir los paths a los directorios que contienen las imágenes como vemos en la imágen** más abajo. Incluye funcionalidades como indicar un documeento CSV con las imágenes por tipo de luz para un análisis más profundo del entrenamiento más adelante.
<p align="center">
  <img width="700" alt="IDP_cargar_imágenes" src="https://github.com/user-attachments/assets/1c330c9b-69ff-4397-b5a0-60efd14d7bad" />
</p>

#### 3️⃣ **Mostrar el análisis**
  IDP hace un análisis de las imágenes incluidas y se puede obtener información de estas usando funciones como: 
  - _show_image_ para mostrar imágenes del dataset cargado.
  - _print_summary_ para mostrar en forma de texto un resumen de la composición del dataset.
  - _graph_summary_ para mostrar los gráficos con esos mismos datos, un ejemplo de esto es la imagen más abajo.

<p align="center">
  <img width="700" alt="grafo_dataset_piccolo" src="https://github.com/user-attachments/assets/dbec55e8-1614-432c-9374-11c4c6fe36c4" />
</p>


#### 4️⃣ **Obtener los dataloaders**
En este último paso obtenemos los dataloaders, una estructura de pytorch que contiene los splits de cada modelo, y es requerida para cargar estos datos en entrenamiento de manera eficiente.

Para obtenerlos, la clase **IDP implementa la función _get_dataloaders_**. Esta recibe parámetros como el tamaño del batch o si utilizar los splits
definidos en el paso de load_dataset. Otros parámetros útiles son: 
- _analize_splits_ para dar información extra sobre los splits y su composición.
- _rand_ para obtener aleatoriamente x cantidad de imágenes para cada split si no queremos entrenar con todos los datos cargados, para ello podemos usar los parámetros de entrada _train_split_, _val_split_ y _test_split_ para determinar el número de imágenes para cada split.

<p align="center">
  <img width="700" alt="IDP_get_dataloaders" src="https://github.com/user-attachments/assets/29230fb3-da3a-47a2-a587-cddd288afb3d" />
<\p>

### 🏃 Train Model

Esta segunda clase es la encargada del entrenamiento de todos los modelos así de la obtención de valiosa información del rendimiento del modelo durante el entrenamiento y otros
detalles que puede ser muy útil a la hora de analizar el modelo y corregir tendencias negativas
El uso de esta clase consta de dos sencillos pasos

#### 1️⃣ Inicializar la clase
**Simplemente indicando el modelo, la loss function y optimizador de nuestra preferencia** como podemos ver en la [imagen](#TRM_ex).

Otros parámetros útiles pueden ser: 
- _eval_pred_ para indicar que quieres evaluar la predicción y ofrece más datos en post entrenamiento como un heatmap con los centros de las bbox objetivo y predichas.
- _meta_path_ para utilizar los metadatos clínicos del dataset en en análisis.

#### 2️⃣ Entrenar
Para ello simplemente llamamos a la función _train_model_ y incluimos parámetros como: el modelo, el número de épocas y los dataloaders obtenidos
anteriormente. Esta implementación podemos verla en la siguiente imagen.

<p align="center" id="TRM_ex">
  <img width="800" alt="TrainModel_uso" src="https://github.com/user-attachments/assets/b49faae3-4fe1-4c52-a1fa-bfb9f890af15" />
</p>


