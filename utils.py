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
import csv

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
    def __init__(self, json_file, image_folder, train_resolution):
        """
        Inicializa la clase con el archivo JSON de COCO y la carpeta local de imágenes.
        
        - json_file: Ruta al archivo JSON de COCO.
        - image_folder: Ruta a la carpeta local donde están almacenadas las imágenes.
        """
        self.json_file = json_file
        self.image_folder = image_folder
        self.json = self._load_json()
        self.image_dict = self._create_image_dict()
        self.shape = (0,0)                      # forma del dataset
        self.train_res = train_resolution       # Resolcuión de las imagenes en train
        # Clases de coco ordenadas por id, 1: person
        self.coco_classes = [
            "unlabeled", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
            "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass",
            "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table",
            "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush", "hair brush", "banner", "blanket", "branch", "bridge", "building-other", "bush",
            "cabinet", "cage", "cardboard", "carpet", "ceiling-other", "ceiling-tile", "cloth", "clothes", "clouds",
            "counter", "cupboard", "curtain", "desk-stuff", "dirt", "door-stuff", "fence", "floor-marble", "floor-other",
            "floor-stone", "floor-tile", "floor-wood", "flower", "fog", "food-other", "fruit", "furniture-other", "grass",
            "gravel", "ground-other", "hill", "house", "leaves", "light", "mat", "metal", "mirror-stuff", "moss",
            "mountain", "mud", "napkin", "net", "paper", "pavement", "pillow", "plant-other", "plastic", "platform",
            "playingfield", "railing", "railroad", "river", "road", "rock", "roof", "rug", "salad", "sand", "sea",
            "shelf", "sky-other", "skyscraper", "snow", "solid-other", "stairs", "stone", "straw", "structural-other",
            "table", "tent", "textile-other", "towel", "tree", "vegetable", "wall-brick", "wall-concrete", "wall-other",
            "wall-panel", "wall-stone", "wall-tile", "wall-wood", "water-other", "waterdrops", "window-blind",
            "window-other", "wood"
        ]

    def _load_json(self):
        with open(self.json_file, 'r') as f:
            return json.load(f)
        
    def get_shape(self):
        return (len(self.json['images']), 3)    # caracteristicas: id, bbox, categoria
    
    def _bbox_COCO2YOLO(self, bbox, img_w, img_h):
        """"
        Esta funcion procesa las bboxes del formato COCO [x, y, w, h] (0-num pixels)
        a formato de salida de YOLO [cx, cy, w, h] normalizado (0-1)
        """
        x, y, w, h = bbox

        # Obtenemos el centro de la bbox
        cx = x + (w / 2)
        cy = y + (h / 2)

        # normalizamos los datos
        cx_norm = cx / img_w
        cy_norm = cy / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        # Devolvemos el formato de YOLO
        return [cx_norm, cy_norm, w_norm, h_norm]
    
    def _bbox_yolo2coco(self, yolo_bbox, img_w, img_h):
        """
        Esta funcion pasa del formato YOLO noramlizado de [cx, cy, w, h]
        al formato COCO en coordenadas de la imagen [x, y, w, h] -> xy esq sup izq
        """
        cx_yolo, cy_yolo, w_yolo, h_yolo= yolo_bbox


        w = int(w_yolo * img_w)
        h = int(h_yolo * img_h)
        cx = cx_yolo * img_w
        cy = cy_yolo * img_h
        x = cx - (w / 2)
        y = cy - (h / 2)

        # Hacemos los números enteros para evitar fallos
        return [int(x), int(y), int(w), int(h)]


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
            # Obtener la ruta local completa de la imagen
            file_name = image['file_name']
            image_path = os.path.join(self.image_folder, file_name)
            
            image_dict[image_id] = {
                'image_id': image_id,
                'image_path': image_path,
                'width': image['width'],
                'height': image['height']
            }

        # obtenemos los datos de 'annotations' ya que no están por el mismo orden
        for image in self.json['annotations']:
            bbox_yolo = self._bbox_COCO2YOLO(image['bbox'], 
                                           image_dict[image_id]['width'], 
                                           image_dict[image_id]['height'])
            image_id = image['image_id']

            # juntamos las anotacions con la informacion de la imagen
            final_dict[image_id] = {
                'image_path': image_dict[image_id]['image_path'],
                'bbox_yolo': bbox_yolo,         # bbox formato YOLO
                'bbox_coco': image['bbox'],     # bbox formato COCO
                'category_id': image['category_id']
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
            'bbox_yolo': [],
            'bbox_coco': [],
            'category_id': []
        }

        # Llenamos el diccionario con los datos de `final_dict`
        for image_id, image_data in self.image_dict.items():
            if image_id in ids:
                data['image_path'].append(image_data['image_path'])
                data['bbox_yolo'].append(image_data['bbox_yolo'])
                data['bbox_coco'].append(image_data['bbox_coco'])
                data['category_id'].append(image_data['category_id'])

        # Ahora creamos el Dataset
        return Dataset.from_dict(data)
        
    def show_image(self, id, pred_bbox=None):
        """
        Esta funcion muestra la imagen con id proporcionado usando una figura de 
        matplotlib, ya que cv2 crashea el kernel.
        Mostramos la imagen id si tenemos una prediccion de YOLO la muestra también
        """
        img_path = self.image_dict.get(id)['image_path']
        coco_x, coco_y, coco_w, coco_h = self.image_dict.get(id)['bbox_coco'] # Formato YOLO
        img_label = self.image_dict.get(id)['category_id']
        img = mpimg.imread(img_path)
        h_img, w_img = img.shape[:2]

        # Creamos la figura para añadir los datos
        fig, ax = plt.subplots(1) 
  
        # Cargamos la imagen en la figura
        ax.imshow(img)

        # Añadimos el dibujo de la bbox
        rect = patches.Rectangle((coco_x, coco_y), coco_w, coco_h, linewidth=1, 
                                edgecolor='r', facecolor="none") 

        text_x = min(coco_x+4, w_img - 50)   # evitamos que el texto se salga de la imagen
        text_y = max(coco_y-9, 9)
        plt.text(text_x, text_y, self.coco_classes[img_label], backgroundcolor='r',
                 color='w', fontname='monospace', size='x-small')
        
        # Add the patch to the Axes 
        ax.add_patch(rect) 
        plt.show()


class ImageDatasetProcessor:
    """
    Esta clase procesa el dataset dado guardandolo como un diccionario de elementos.
    Ofrece herramientas y funciones para a partir de este diccionario salvar, modificar
    o obtener información relevante de el dataset.
    Esta clase guarda el dataset procesado en un json en el path dado
    """

    def __init__(self, target_resolution=(256, 256),
                 dataset_name=None, json_path=None,):
        """
        Esta funcion inicializa el dataset creando un diccionario a partir del 
        json del dataset dado o en su defecto, procesa el dataset y lo crea.

        - target_resolution: reolución del reescalado final de la imagen para 
            introducirla como tensor en el modelo, se usa para mostrar datos
        - format_doc: documento de texto con el tipo de luz de cada imagen del ds
        - dataset_name: nombre del dataset utilizado para procesar el dataset 
            de forma personalizada, admitidos: "Piccolo"
        - json_path: path al json del dataset, si no se aprota entonces se crea
        - polyp_paths: path a las imagenes de polipos, uno o varios directorios
        - mask_paths: path a las imagenes de macaras, uno o varios directorios
        - void_paths: path a las imagenes de void , uno o varios directorios
        """

        # contexto que nos da el usuario
        self.target_resolution = target_resolution
        self.json_path = json_path

        # diccionario de imágenes y su formato
        self.dict = {}
        self.format = {
            "path": "",
            "mask_path": "",
            "void_path": "None",
            "size": (0, 0),
            "light_type": "Unknown",    # WL, NBI, BLI, ...
            "bbox": (0, 0, 0, 0),       # esquina sup izq normalizado (x, y, w, h)
            "split": "",                # train, test, validation
        }

        # Creamos el diccionario si no existe el json
        if json_path is None or not os.path.isfile(json_path):
            self.ds_dict = self._create_image_dict(dataset_name)
        # else:   # cargamos el diccionario del json


    def load_dataset(self, polyps_path, masks_path, void_path=None, 
                     split="None", dir_light_type=None, light_csv=None):
        """
        Dado el directorio de las imagenes del dataset crea un diccionario con
        las imagenes de los polipos con informacion relevante como sus máscaras 
        , void si los hubiera, tamaño, tipo de luz etc.

        - polyps_path: directorio con imágenes de pólipos
        - masks_path: directorio con las máscaras de los pólipos
        - voids_path voids de las imágenes si los hubiera
        - split: a que split pertenece el directorio de imagenes
        - light_type: el tipo de luz de las imágenes del directorio
        - light_type_file: ruta al csv con la clasificacion de luz de las imagenes
        nombre_img,tipo_de_luz
        """
        image_dict = {}

        # Primero guardamos una lista ordenada con todos los polypos y mascaras
        polyp_list = sorted(os.listdir(polyps_path))
        mask_list = sorted(os.listdir(masks_path))
        void_list = sorted(os.listdir(void_path)) if void_path else None

        # cargamos el csv si lo hubiera y la información
        light_types = {}
        if light_csv is not None:
            with open(light_csv, newline='', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter=';')
                # formato "nombre imagen": tipo de luz
                for row in reader:
                    light_types[row[0]] = row[1].strip()

        # guardamos los datos del directorio dado con el formato del diccionario
        for i, polyp in enumerate(polyp_list):
            img_name = os.path.basename(polyp)
            img_path = os.path.join(polyps_path, polyp)
            mask_path = os.path.join(masks_path, mask_list[i])
            void_path = os.path.join(void_path, void_list[i]) if void_list else None
            light_type = dir_light_type or light_types.get(img_name, "Unknown")

            # primero obtenemos el formato de la imagen
            with Image.open(img_path) as img:
                img_size = img.size()

            # Ahora obtenemos la bbox
            with Image.open(mask_path) as mask:
                bbox = self._bbox_from_mask(mask)
            
            # Ahora guardamos todos los datos en el diccionario
            image_dict[img_name] = self.format.copy()
            image_dict[img_name]["path"] = img_path
            image_dict[img_name]["mask_path"] = mask_path
            image_dict[img_name]["size"] = img_size
            image_dict[img_name]["light_type"] = light_type
            image_dict[img_name]["bbox"] = bbox
            image_dict[img_name]["split"] = split

        return image_dict
    
    def _bbox_from_mask(mask):
        """
        Dada una máscara devuelve las coordenadas de la bbox con el formato de
        COCO: (x, y, anchura, altura) / x, y es la esquina superior izquierda
        todo ello normalizado.
        """
        mask_array = np.array(mask)  # Convertir la máscara en array NumPy
        rows, cols = np.where(mask_array > 0)  # Encontrar píxeles no negros

        if len(rows) == 0 or len(cols) == 0:
            return (0, 0, 0, 0)  # Si la máscara está vacía

        # Coordenadas mínimas y máximas
        min_x, max_x = cols.min(), cols.max()
        min_y, max_y = rows.min(), rows.max()

        # Ancho y alto
        width = max_x - min_x
        height = max_y - min_y

        # Normalizar según tamaño de la imagen
        img_width, img_height = mask.size
        bbox = (min_x / img_width, min_y / img_height, width / img_width, height / img_height)

        return bbox
    

    # return a list of all elements of the given paths
    def _ls_recursive(paths):
        my_list = []

        for path in paths:
            my_list.append(os.listdir(path))
        
        return sorted(my_list)
