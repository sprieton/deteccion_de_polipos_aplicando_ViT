"""
En este .py he ido creando herramientas necesarias para procear datos entre ellas se encuentra:
# DatasetExplorer

Este módulo proporciona la clase `DatasetExplorer`, diseñada para facilitar el análisis exploratorio de conjuntos de datos de colonoscopias, 
especialmente aquellos que contienen imágenes de pólipos y sus máscaras asociadas.

### Funcionalidad:
- Carga y organiza imágenes de un dataset en función de su tipo (pólipos, máscaras, márgenes).
- Extrae y analiza características relevantes como resolución, número de canales, brillo y contraste.
- Calcula estadísticas clave sobre la distribución de imágenes y máscaras dentro del dataset.
- Genera visualizaciones para facilitar la interpretación de los datos, incluyendo histogramas y mapas de calor.

# COCODataProcessor

Esta clase proporciona la herramienta `COCODataProcessor`, diseñada para facilitar la extracción y organización de datos del dataset COCO (Common Objects in Context),
especialmente para trabajar con imágenes y sus anotaciones asociadas. Está orientada a crear un diccionario que asocie las imágenes con las coordenadas de los bounding boxes y otros datos relevantes.

### Funcionalidad:
- Carga y organiza los datos del archivo JSON de anotaciones COCO (en formato `.json`).
- Extrae información clave sobre cada imagen, como su `id_imagen`, `filename`, `altura`, `anchura`, y la ruta local de la imagen en la carpeta proporcionada.
- Procesa las anotaciones asociadas a cada imagen, extrayendo datos relevantes como `clase`, `bbox` (bounding box) y `segmentación`.
- Vincula las imágenes con su ruta local en la carpeta de imágenes, lo que permite una fácil asociación de los datos con las imágenes reales.
- Organiza toda la información en un diccionario estructurado, lo que facilita su uso en modelos de aprendizaje automático u otras aplicaciones de análisis de datos.
- Permite almacenar los datos procesados en un nuevo archivo JSON para su posterior uso o análisis.

# Ejemplo de uso:
1. Instanciando la clase con el archivo JSON de anotaciones y la carpeta de imágenes:
   processor = COCODataProcessor(json_file, image_folder)
2. Procesando las anotaciones y obteniendo el diccionario con los datos:
   processed_data = processor.process_annotations()
3. Guardando el resultado en un archivo JSON:
   with open('processed_data.json', 'w') as f:
       json.dump(processed_data, f, indent=4)

"""


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches 
import seaborn as sns
import json
from datasets import Dataset
from PIL import Image
import torch

class DatasetExplorer:
    def __init__(self, target_resolution=(256, 256), format_doc=None, dataset_name=None):
        """
        Inicialización de la clase.
        - target_resolution: resoluición objetivo sirve para definir
            la resolución de las máscaras.
        - format_doc: documento de formato sirve para obtener
            el tipo de luz que tiene cada frame.
        - dataset_name: sirve para cargar los datos teniendo en cuenta el dataset
            ya que cada uno tiene un formato de datos distinto
        """
        # Imágenes del dataset
        self.polyp_img = []         # Paths de imágenes de pólipos
        self.bin_img = []           # Paths de máscaras binarias si existen
        self.void_img = []          # Paths de imágenes de márgenes si existen

        # Información sobre el dataset
        self.format_counts = {}         # Número de imágenes por formato: WLI, NBI ...
        self.function_counts = {}       # Número de imágenes por funcion; test, train ...
        self.resolution_counts = {}     # Conteo de imágenes por resolución
        self.channel_counts = {}        # Conteo de imágenes por número de canales
        self.brightness = []            # Brillo de las imagenes
        self.contrast = []              # Contraste de las imagenes

        # heatmap con la distribución de las máscaras
        self.mask_heatmap = np.zeros(target_resolution, dtype=np.float32)    
        self.mask_heatmap_shape = target_resolution

        self.mean_masks_percentage = 0      # media de porcentage de ocupación de las máscaras en la imagen
        if format_doc is not None:          # documento con la especificacion de formato de las imágenes usado por: piccolo
            self.format_doc = open(format_doc, 'r')
        else:
            self.format_doc = None
        self.dataset_name = dataset_name


    def path_load_images(self, directory_path, img_type, img_func, img_format):
        """
        Carga los paths de las imágenes de un directorio dado.
        Ordena estas imágenes para poder relacionarlas con sus máscaras
        """
        image_paths = []
        file_names = sorted(os.listdir(directory_path))

        for file in file_names:
            file_path = os.path.join(directory_path, file)
            img = cv2.imread(file_path)
            if img is not None:
                image_paths.append(file_path)
                self._update_stats(img, img_type, img_func, file, img_format)  # Actualiza estadísticas de la imagen
            else:
                print("Error cargando {}".format(file_path))
                
        return image_paths

    def _update_stats(self, img, img_type, img_func, file_name, img_format=None):
        """
        Actualiza la información de imágenes del dataset : 
            - Resolución: tipos de resoluciones y cantidad de imagenes en ese formato
            - Número de canales: distribucion de imagenes por numero de canales
            - Número de imagen por tipo: clasificacion de la imagenes por tipo de Luz
            - Distribucion de las máscaras:
            Donde se encuentran las máscaras distribuidas en las imágenes
            y el porcentaje de la imágen que cubre el pólipo
        """
        # Revisamos los canales de la imágen, shape[2] solo tiene valor si son 3 canales
        channels = str(img.shape[2]) + f"_{img_type}" if len(img.shape) > 2 else f"1_{img_type}"
        if channels not in self.channel_counts:
            self.channel_counts[channels] = 0
        self.channel_counts[channels] += 1

        # Actualizar tipo de imagen, resolución y función por formato solo para imágenes tipo polyp
        if img_type == "polyp":
            # Actualizar resolución
            resolution = f"{img.shape[1]}x{img.shape[0]}"
            if resolution not in self.resolution_counts:
                self.resolution_counts[resolution] = 0      # creamos una nueva entrada en el diccionario
            self.resolution_counts[resolution] += 1

            # Actualizar tipo de imagen por función
            function = f"{img_func}"
            if function not in self.function_counts:
                self.function_counts[function] = 0
            self.function_counts[function] += 1

            # Actualizar formato de imagen si no lo tenemos ya
            format = img_format or f"{self._obtain_img_format(file_name, self.dataset_name)}"
            if format not in self.format_counts:
                self.format_counts[format] = 0
            self.format_counts[format] += 1

            # Actualizar el brillo de la imágen
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.brightness.append(np.mean(gray))   # como de brillante es el valor de los pixeles

            # Actualizar el contraste de la imágen
            self.contrast.append(np.std(gray))  # El contraste es la desviación estándar de los píxeles

        # Actualizar los datos de las máscaras
        if img_type == "mask":
            # primero reducimos a un solo canal para poder trabajar la imágen
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # antes de reescalar calculamos el porcentaje de imágen cubierto por la máscara
            mask_pixels = np.sum(grey == 255)  # Píxeles de la máscara
            total_pixels = grey.size
            percentage = (mask_pixels / total_pixels) * 100

            self.mean_masks_percentage += percentage

            # ahora reescalamos a la calidad especificada, formato (alto, ancho)
            mask = cv2.resize(grey,
                              (self.mask_heatmap_shape[1], self.mask_heatmap_shape[0]),
                              interpolation=cv2.INTER_LINEAR)

            # Normalizar los valores de la máscara entre 0 y 1
            mask = mask / 255.0

            # Sumar al heatmap
            self.mask_heatmap += mask

    def _obtain_img_format(self, file_name, dataset=None):
        """
        Devuelve el tipo de imagen del que es el dataset, teniendo en cuenta
        el dataset que estamos cargando
        """
        img_light = "UNKNOWN"

        # Si no tenemos fichero con el que conocer los datos
        if self.format_doc is None:
            return img_light

        if dataset is not None:   # tenemos un dataset en concreto
            if dataset == "Piccolo":    # formato del dataset Piccolo
                for line in self.format_doc:    # buscamos en cada línea
                    if file_name in line:        # si encontramos la imagen
                        # obtenemos su tipo tal que el formato es: nombre_imagen;Tipo_de_luz
                        img_light = line.split(";")[1].strip()
                        continue

        # si no hemos encontrado el fichero
        if img_light == "UNKNOWN":
            print(f"sin datos para: {file_name}")

        # Reinicia el puntero al principio del archivo para volver a buscar
        self.format_doc.seek(0)
        return img_light
    

    def load_dataset(self, path, img_type, img_function, img_format=None):
        """
        Carga el conjunto de imágenes dado y actualiza las estadísticas.
        """

        if img_type == "polyp":
            self.polyp_img.extend(self.path_load_images(path, img_type, img_function, img_format))
        
        if img_type == "mask":
            self.bin_img.extend(self.path_load_images(path, img_type, img_function, img_format))

        if img_type == "void":
            self.void_img.extend(self.path_load_images(path, img_type, img_function, img_format))
        
        print("Directorio \"{}\" cargado con éxito".format(path))

    def print_summary(self):
        """
        Imprime un resumen de las estadísticas del dataset.
        """
        print("Total imágenes:")
        print(f"\t- Polyp: {len(self.polyp_img)}")
        print(f"\t- Mask: {len(self.bin_img)}")
        print(f"\t- Void: {len(self.void_img)}")

        print("Composición del dataset:")
        for dictionary in [
                self.format_counts, 
                self.function_counts, 
                self.resolution_counts,
                self.channel_counts]:
            if dictionary == self.format_counts:
                print("Formatos:")
            elif dictionary == self.function_counts:
                print("Función:")
            elif dictionary == self.resolution_counts:
                print(f"Resoluciónes: total distintas resoluciones {len(self.resolution_counts)}")
            else:
                print("Canales:")
            for data, num in dictionary.items():
                print(f"\t{data}: {num}", end="")
            print("\n")

        self.mean_masks_percentage = self.mean_masks_percentage / len(self.bin_img)
        print(f"Volumen medio de los pólipos respecto a la imagen:\t{self.mean_masks_percentage}%")
    
    def graph_summmary(self):
        # Configuración del estilo de los gráficos
        sns.set(style="whitegrid")

        # Crear gráficos
        fig, axs = plt.subplots(4, 2, figsize=(10, 8))

        # Graficamos los diagramas
        charts = [
            # Gráfico 1: Distribución de las imágenes por función
            (self.function_counts, axs[0, 0], 'División de imágenes del dataset', 'Número de Imágenes'),
            # Gráfico 2: Composición del dataset por tipo de imágen
            (self.format_counts, axs[0, 1], 'Composición del dataset por tipo de imágen', 'Número de Imágenes'),
            # Gráfico 3: Tipos de resoluciones en las imágenes del dataset
            (self.resolution_counts, axs[1, 0], 'Tipos de resoluciones en las imágenes del dataset', 'Número de Imágenes'),
            # Gráfico 4: Número de canales por tipo de imágen
            (self.channel_counts, axs[1, 1], 'Número de canales por tipo de imágen', 'Número de Imágenes')
        ]

        for data, ax, title, ylabel in charts:
            ax.set_ylabel(ylabel)
            ax.bar(data.keys(), data.values(), color=['blue', 'orange', 'green'])
            ax.set_title(title)

        # Graficamos los histogramas
        hist = [
            # Gráfico 5: Histograma del brillo en las imágenes
            (self.brightness, axs[2, 0], 'Brillo de los frames', 'Número de Imágenes'),
            # Gráfico 6: Histograma del contraste en las imágenes
            (self.contrast, axs[2, 1], 'Contraste de los frames', 'Número de Imágenes')
        ]
        
        for data, ax, title, ylabel in hist:
            ax.hist(data, bins=20, color='darkgreen')
            ax.set_title(title)
            ax.set_ylabel(ylabel)


        # Gráfico 7: heatmap de distribución de las máscaras
        if self.mask_heatmap is not None:
            sns.heatmap(
                self.mask_heatmap,
                cmap="crest",
                ax=axs[3, 0],
                cbar=True,
                xticklabels=False,
                yticklabels=False
            )
            axs[3, 0].set_title('Distribución de las Máscaras (Heatmap)')

        axs[3, 1].axis('off')

        # Ajustar el layout
        plt.tight_layout()
        plt.show()

class COCODataProcessor:
    def __init__(self, json_file, image_folder):
        """
        Inicializa la clase con el archivo JSON de COCO y la carpeta local de imágenes.
        
        - json_file: Ruta al archivo JSON de COCO.
        - image_folder: Ruta a la carpeta local donde están almacenadas las imágenes.
        """
        self.json_file = json_file
        self.image_folder = image_folder
        self.json = self._load_json()
        self.image_dict = self._create_image_dict()
        self.shape = (0,0)      # forma del dataset
        # Clases de coco ordenadas por id, 1: person
        self.coco_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", 
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def _load_json(self):
        with open(self.json_file, 'r') as f:
            return json.load(f)
        
    def get_shape(self):
        return (len(self.json['images']), 3)    # caracteristicas: id, bbox, categoria

    def _create_image_dict(self):
        """
        Crea un diccionario de imágenes donde la clave es el `id` de la imagen
        y el valor es un diccionario con información sobre la imagen.
        """
        final_dict = {}
        image_dict = {}

        # obtenemos la información de la imagen
        for image in self.json['images']:
            image_id = image['id']
            file_name = image['file_name']
            # Obtener la ruta local completa de la imagen
            image_path = os.path.join(self.image_folder, file_name)
            
            image_dict[image_id] = {
                'image_id': image_id,
                'image_path': image_path
            }

        # obtenemos los datos de 'annotations' ya que no están por el mismo orden
        for image in self.json['annotations']:
            bbox = image['bbox']
            category_id = image['category_id']
            image_id = image['image_id']

            final_dict[image_id] = {
                # juntamos las anotacions con la informacion de la imagen
                'image_path': image_dict[image_id]['image_path'],
                # informacion de las anotaciones
                'bbox': bbox,
                'category_id': category_id
            }

        return final_dict
    
    def dataset_from_dict_ids(self, ids):
        """
        Esta funcion devuelve un dataset con los ids de imagenes dados.
        Para ello usa la funcion Dataset.from_dict
        """
        # Convertimos el diccionario en un formato adecuado para `from_dict()`
        # Cada clave será una lista de valores para cada campo
        data = {
            'image_path': [],
            'bbox': [],
            'category_id': []
        }

        # Llenamos el diccionario con los datos de `final_dict`
        for image_id, image_data in self.image_dict.items():
            if image_id in ids:
                data['image_path'].append(image_data['image_path'])
                data['bbox'].append(image_data['bbox'])
                data['category_id'].append(image_data['category_id'])

        # Ahora creamos el Dataset
        return Dataset.from_dict(data)
    
    def show_image(self, id):
        """
        Esta funcion muestra la imagen con id proporcionado usando una figura de 
        matplotlib, ya que cv2 crashea el kernel
        """
        img_path = self.image_dict.get(id)['image_path']
        x, y, width, height = self.image_dict.get(id)['bbox'] # xy esquina inferior izquierda
        img_label = self.image_dict.get(id)['category_id']
        img = mpimg.imread(img_path)

        # Creamos la figura para añadir los datos
        fig, ax = plt.subplots(1) 
  
        # Cargamos la imagen en la figura
        ax.imshow(img)
        
        # Añadimos el dibujo de la bbox
        rect = patches.Rectangle((x, y), width, height, linewidth=1, 
                                edgecolor='r', facecolor="none") 
    
        plt.text(x+4, y-10, self.coco_classes[img_label-1], backgroundcolor='r',
                 color='w', fontname='monospace', size='x-small')
        
        # Add the patch to the Axes 
        ax.add_patch(rect) 
        plt.show()
