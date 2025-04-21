"""
Este módulo contiene herramientas esenciales para el procesamiento, análisis y entrenamiento 
de modelos en datasets de imágenes médicas, con un enfoque particular en colonoscopias 
y detección de pólipos. Incluye clases y funciones para explorar datasets, preparar datos, 
procesar anotaciones, entrenar modelos y visualizar resultados.

### Contenido principal:

- `ImageDatasetProcessor`: gestiona datasets personalizados, cargando imágenes y máscaras, 
  extrayendo estadísticas, generando divisiones (train/val/test) y creando dataloaders.

- `TrainModel`: clase principal para entrenar modelos de detección de objetos con PyTorch, 
  monitorizando métricas como IoU y pérdida, y visualizando los resultados del entrenamiento.
  Usa dataloaders con el formato ImageDatasetProcessor

- Funciones auxiliares (`bbox_corn2cent`, `bbox_cent2corn`, etc.): conversión entre formatos 
  de bounding boxes y cálculo del IoU.

Este archivo es una base sólida para construir un pipeline completo de procesamiento 
y entrenamiento con imágenes médicas segmentadas.
"""



import os
import cv2
import json
import csv
import json
import random
import torch
import pynvml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches 
import seaborn as sns
from datasets import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
import torchvision.transforms as transforms

class ImageDatasetProcessor:
    """
    Esta clase procesa el dataset dado guardandolo como un diccionario de elementos.
    Ofrece herramientas y funciones para a partir de este diccionario salvar, modificar
    o obtener información relevante de el dataset.
    Esta clase guarda el dataset procesado en un json en el path dado
    """

    def __init__(self, target_resolution=(256, 256),
                 dataset_name=None, json_path=None):
        """
        Esta funcion inicializa la clasee ImageDatasetProcessor que se debe usar 
        de estas dos maneras:
        1- Nuevo dataset, creamos la clase con json_path= a la dirección y nombre
        del json donde guardaremos los datos procesados. Para luego usar load_dataset
        para cargar todas los conjuntos de datos deseados.
        2- Dataset ya procesado, al crear la clase y aportar el json esta función
        ya carga la información directamente de ahí.

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
        self.target_resolution = target_resolution      # ancho x alto
        self.json = json_path

        # Información sobre el dataset
        self.resolution_counts = {}     # Conteo de imágenes por resolución
        self.light_counts = {}          # Número de imágenes por formato: WLI, NBI ...
        self.split_counts = {}          # Número de imágenes por funcion; test, train ...
        self.channel_counts = {}        # Conteo de imágenes por número de canales
        self.brightness = []            # Brillo de las imagenes
        self.contrast = []              # Contraste de las imagenes
        self.polyp_centers = []         # Centros de los pólipos
        # suma de la dist euclidea del centro del ṕolipo respecto al centro img
        self.sum_mask_eucl_dist2center = 0
        # heatmap con la distribución de las máscaras
        self.mask_heatmap = np.zeros(target_resolution, dtype=np.float32)
        # suma de porcentage de ocupación de las máscaras en la imagen
        self.sum_masks_percentage = 0


        # diccionario de imágenes y su formato
        self.dict = {}
        self.format = {
            "path": "",
            "mask_path": "",
            "void_path": "None",
            "size": (0, 0),
            "light_type": "Unknown",    # WL, NBI, BLI, ...
            "bbox": (0, 0, 0, 0),       # formato center (cx, cy, w, h)
            "split": "None",            # train, test, validation, no split
        }

        # json ya existe, por lo que sencillamente lo cargamos
        if os.path.isfile(self.json):
            self._load_from_json()


    def load_dataset(self, polyps_path, masks_path, voids_path=None, 
                     split="None", dir_light_type=None, light_csv=None):
        """
        Dado el directorio de las imagenes del dataset crea un diccionario con
        las imagenes de los polipos con informacion relevante como sus máscaras 
        , void si los hubiera, tamaño, tipo de luz etc. y actualiza con esta
        información el diccionario del dataset dado, además guarda los datos
        procesados en un json si este ha sido aportado

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
        void_list = sorted(os.listdir(voids_path)) if voids_path else None

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
            void_path = os.path.join(voids_path, void_list[i]) if void_list else None
            light_type = dir_light_type or light_types.get(img_name, "Unknown")

            # primero obtenemos el formato de la imagen
            img = Image.open(img_path)
            img_size = img.size

            # Ahora obtenemos la bbox
            mask = Image.open(mask_path)
            bbox = self._bbox_from_mask(mask)

            # actualizamos las estadísticas del dataset
            self._update_stats(img, mask, void_path, light_type, split, img_name)
            img.close()     # cerramos las imágenes
            mask.close()
            
            # Ahora guardamos todos los datos en el diccionario
            image_dict[img_name] = self.format.copy()
            image_dict[img_name]["path"] = img_path
            image_dict[img_name]["mask_path"] = mask_path
            image_dict[img_name]["void_path"] = void_path
            image_dict[img_name]["size"] = img_size
            image_dict[img_name]["light_type"] = light_type
            image_dict[img_name]["bbox"] = bbox
            image_dict[img_name]["split"] = split
            # print(f"Imagen procesada {img_name}: {i}")
        
        # finalmente guardamos los nuevos datos en el diccionario
        self.dict.update(image_dict)

        # y sobreescribimos el json con la nueva información si existe
        if self.json is not None:
            self._save_on_json()

    
    def get_dataloaders(self, batch_size, use_premade_splits=False, rand=False,
                        train_split=0, val_split=0, test_split=0):
        """
        Función que divide el diccionario del dataset los tres conjuntos de train
        validation y test.
        
        - use_premade_splits: puedes usar el conjunto prehecho del datset si lo hay
        o indicar el porcentaje de cada split.
        - rand: si quieres mezclar aleatoriamente las imágenes del dataset antes del split
        - train_split: num elementos del dataset para el conjunto de entrenamiento
        - val_split: num elementos del dataset para el conjunto de validacion
        - test_split: num elementos del dataset para el conjunto de test
        """

        train_ids = []
        val_ids = []
        test_ids = []
        
        # Comprovaciones de que se llama a la función correctamente
        first_img = list(self.dict)[0]
        if use_premade_splits and self.dict[first_img]["split"] == "None":
            print("No hay un split prehecho para crear los dataloaders!")
            return None, None, None
        
        if train_split + val_split + test_split > len(self.dict):
            print(f"Splits indiccados superan el número de elementos del dataset: {len(self.dict)}")

        # Primero elegimos las imágenes de los splits
        image_ids = list(self.dict.keys())  # obtenemos los nombres de las imágenes

        # Usamos el split de base
        if use_premade_splits:
            for img, data in self.dict.items():  # recorremos los valores
                if data["split"] == "train":
                    train_ids.append(img)
                elif data["split"] == "validation":
                    val_ids.append(img)
                else:
                    test_ids.append(img)
        # hacemos el split a mano, aleatorio o no
        else:
            if rand:
                # Obtenemos la cantidad de imagenes con las que queremos trabajar aleatoriamente
                selected_ids = np.random.choice(image_ids, 
                                                size=train_split+val_split+test_split, 
                                                replace=False)
                suffle=True
            else:   # cogemos las primeras del dataset
                selected_ids = image_ids[:test_split+val_split+train_split]
                suffle=False
            
            # Ahora, separamos los conjuntos
            train_val_ids, test_ids = train_test_split(selected_ids, 
                                                    test_size=test_split, 
                                                    suffle=suffle)
            # Separamos validation de train y terminamos con los 3 conjuntos
            train_ids, val_ids = train_test_split(train_val_ids, test_size=val_split, 
                                                suffle=suffle)

        # Obtenemos los dataset conteniendo cada uno las imagenes seleccionadas
        dtrain = self._dataset_from_dict_ids(train_ids)
        dval = self._dataset_from_dict_ids(val_ids)
        dtest = self._dataset_from_dict_ids(test_ids)

        # Finalmente creamos los dataLoaders de las imágenes de cada split
        # suffle para mezclar los datos en cada época 
        # pin_memory para acelerar CPU->GPU
        train_loader = DataLoader(dtrain, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(dval, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(dtest, batch_size=batch_size, shuffle=True, pin_memory=True)

        return train_loader, val_loader, test_loader
    

    def print_summary(self):
        """
        Imprime un resumen de las estadísticas del dataset.
        """
        print(f"Total imágenes: {len(self.dict)}")

        print("Composición del dataset:")
        for dictionary in [
                self.resolution_counts,
                self.light_counts, 
                self.split_counts, 
                self.channel_counts]:
            if dictionary == self.resolution_counts:
                print(f"Resoluciónes: total distintas resoluciones {len(self.resolution_counts)}")
            elif dictionary == self.light_counts:
                print("Tipos de luz:")
            elif dictionary == self.split_counts:
                print("Splits:")
            else:
                print("Canales:")
            for data, num in dictionary.items():
                print(f"\t{data}: {num}", end="")
            print("\n")

        mean_masks_percentage = self.sum_masks_percentage / len(self.dict)
        print(f"Volumen medio de los pólipos respecto a la imagen:\t{mean_masks_percentage}%")
        mean_mask_eucl_dist2center = self.sum_mask_eucl_dist2center / len(self.dict)
        print(f"Distancia media del centro del pólipos al centro de la imagen:\t{mean_mask_eucl_dist2center}px")
    
    def graph_summmary(self):
        # Configuración del estilo de los gráficos
        sns.set(style="whitegrid")

        # Crear ventana con gráficos
        fig, axs = plt.subplots(4, 2, figsize=(10, 8))


        # Graficamos los diagramas
        charts = [
            # Gráfico 1: Distribución de las imágenes por split
            (self.split_counts, axs[0, 0], 'División de imágenes del dataset', 'Número de Imágenes'),
            # Gráfico 2: Composición del dataset por tipo de luz
            (self.light_counts, axs[0, 1], 'Composición del dataset por tipo de luz', 'Número de Imágenes'),
            # Gráfico 3: Tipos de resoluciones en las imágenes del dataset
            (self.resolution_counts, axs[1, 0], 'Tipos de resoluciones en las imágenes del dataset', 'Número de Imágenes'),
            # Gráfico 4: Número de canales por tipo de imágen
            (self.channel_counts, axs[1, 1], 'Formato de las imágenes', 'Número de Imágenes')
        ]

        for data, ax, title, ylabel in charts:
            ax.set_ylabel(ylabel)
            ax.bar(data.keys(), data.values(), color=['blue', 'green', 'orange'])
            ax.set_title(title)

        # Graficamos los histogramas
        hist = [
            # Gráfico 5: Histograma del brillo en las imágenes
            (self.brightness, axs[2, 0], 'Brillo de los frames', 'Número de Imágenes'),
            # Gráfico 6: Histograma del contraste en las imágenes
            (self.contrast, axs[2, 1], 'Contraste de los frames', 'Número de Imágenes')
        ]
        
        for data, ax, title, ylabel in hist:
            ax.hist(data, bins=20, color='forestgreen')
            ax.set_title(title)
            ax.set_ylabel(ylabel)


        # Gráfico 7: heatmap de distribución de las máscaras
        sns.heatmap(self.mask_heatmap, cmap="crest", ax=axs[3, 0], cbar=True,
                    xticklabels=False, yticklabels=False)
        axs[3, 0].set_title('Distribución de las máscaras (Heatmap)')
        # Gráfico 8: muestra de los centros de los pólipos
        cx, cy = zip(*self.polyp_centers)       # dos listas con coordenadas
        axs[3, 1].scatter(cx, cy, c='lightblue', alpha=0.5, s=10)
        axs[3, 1].set_title('Distribución de los centros de las máscaras (Heatmap)')
        axs[3, 1].set_xlim([0, self.target_resolution[0]])  # Ajustamos los ejes para ser como la imágen
        axs[3, 1].set_ylim([0, self.target_resolution[1]]) 

        # Ajustar el layout
        plt.tight_layout()
        plt.show()

    
    def show_image(self, id):
        """
        Esta funcion muestra la imagen con la bbox asociada
        Mostramos la imagen id si tenemos una prediccion de YOLO la muestra también
        """
        img_path = self.dict[id]['path']
        img_w, img_h = self.dict[id]['size']
        x, y, w, h = bbox_cent2corn(self.dict[id]['bbox'], img_w, img_h)
        img = mpimg.imread(img_path)

        print(f"Imagen {id}\tbbox: {(x, y, w, h)}")

        # Creamos la figura para añadir los datos
        fig, ax = plt.subplots(1) 
  
        # Cargamos la imagen en la figura
        ax.imshow(img)

        # Añadimos el dibujo de la bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                edgecolor='cyan', facecolor="none") 
        
        # Add the patch to the Axes 
        ax.add_patch(rect) 
        plt.show()
    

###########################    FUNCIONES PRIVADAS    ###########################


    def _dataset_from_dict_ids(self, ids):
        """
        Esta funcion devuelve un dataset con los ids de imagenes dados.
        Para ello usa la funcion Dataset.from_dict
        """
        # Convertimos el diccionario en un formato adecuado para `from_dict()`
        # Cada clave será una lista de valores para cada campo
        data = {
            'path': [],     # input, imagen del dataset
            'bbox': []      # output la bbox en formato center (cx,cy,w,h)
        }

        # Llenamos el diccionario con los datos de `final_dict`
        for image_id, image_data in self.dict.items():
            if image_id in ids:
                data['path'].append(image_data['path'])
                data['bbox'].append(image_data['bbox'])

        # Ahora creamos el Dataset
        return Dataset.from_dict(data)


    def _update_stats(self, img, mask, void_path, light_type, split, img_name):
        """
        Actualiza las estadísticas con informacion util del dataset: 
            - Resolución: tipos de resoluciones y cantidad de imagenes en ese formato
            - Número de canales: distribucion de imagenes por numero de canales
            - Número de imagen por tipo de luz
            - Distribucion de las máscaras:
            Donde se encuentran las máscaras distribuidas en las imágenes
            y el porcentaje de la imágen que cubre el pólipo
        """
        # 1️⃣- Resolución de la imágen
        resolution = f"{img.size[1]}x{img.size[0]}"
        self._update_stat_dict(self.resolution_counts, resolution)

        # 2️⃣- Tipo de luz de la imágen
        self._update_stat_dict(self.light_counts, light_type)

        # 3️⃣- Split al que pertenece la imagen
        self._update_stat_dict(self.split_counts, split)

        # 4️⃣- Número de canales de PIL por imágen de las tres imágenes
        chan_type_img = f"polyp_{img.mode}"         # polyp
        self._update_stat_dict(self.channel_counts, chan_type_img)
        chan_type_mask = f"mask_{mask.mode}"         # mask
        self._update_stat_dict(self.channel_counts, chan_type_mask)
        if void_path is not None:                   # void si lo hay
            with Image.open(void_path) as void_img:
                chan_type_void = f"void_{void_img.mode}"         # mask
                self._update_stat_dict(self.channel_counts, chan_type_void)

        # 5️⃣- Obtenemos el brillo de la imagen
        # es lo mismo que la media de la escala de grises
        gray_img = img.convert("L")    # L es escala de grises de PIL
        gray_pixels = np.array(gray_img)
        self.brightness.append(np.mean(gray_pixels))

        # 6️⃣- Contraste de la imágen
        # El contraste es la desviación estándar de los píxeles
        self.contrast.append(np.std(gray_pixels))

        # 7️⃣- Porcentaje de ocupación del pólipo en pantalla
        gray_img = mask.convert("L")    # L es escala de grises de PIL
        gray_pixels = np.array(gray_img)
        mask_pixels = np.sum(gray_pixels==255)      # cuántos pixeles son pólipo
        percentage = (mask_pixels / gray_pixels.size) * 100
        self.sum_masks_percentage += percentage

        # 8️⃣- Actualizar el heatmap de disposición de los pólipos respecto 
        # a la imágen
        mask_resized = mask.resize(self.target_resolution, resample=Image.NEAREST)
        mask_norm = np.array(mask_resized) / 255.0
        self.mask_heatmap += mask_norm.transpose()  # np cambia a alto x anchoS

        # 9️⃣- Actualizar la desviación del centro del ṕolipo
        mask_cx, mask_cy = self._get_mask_center(np.array(mask_resized), img_name)
        # guardamos la desviación respecto del centro
        img_cx = self.target_resolution[0]/2
        img_cy = self.target_resolution[1]/2

        # distancia euclídea al centro
        dist = np.sqrt((img_cx - mask_cx)**2 + (img_cy - mask_cy)**2)
        self.sum_mask_eucl_dist2center += dist

        # 🔟- Guardamos el centro del pólipo para represntarlo
        self.polyp_centers.append((mask_cx, mask_cy))


    def _update_stat_dict(self, dict, data):
        """
        Actualiza el diccionario de estadísticas dado, con el dato "data"
        """
        if data not in dict:
            dict[data] = 0      # creamos una nueva entrada en el diccionario
        dict[data] += 1


    def _get_mask_center(self, mask, img_name):
        """
        Obtiene el centro ponderado de la máscara dada, esto es la media de
        los valores de la máscara, devuelve los dos centros
        """
        # obtenemos las coordenadas de los píxeles de la máscara
        y, x = np.where(mask > 0)

        if len(x) == 0 and len(y) == 0:
            cx = self.target_resolution[0]
            cy = self.target_resolution[1]
            # print(f"image {img_name} máscara vacía")
        else:
            # devolvemos el centro de la máscara
            cx = round(np.mean(x)) # centro redondeado
            cy = round(np.mean(y)) # centro redondeado

        return cx, cy

    
    def _bbox_from_mask(self, mask):
        """
        Dada una máscara devuelve las coordenadas de la bbox con el formato de
        YOLO: (cx, cy, anchura, altura) todo ello normalizado.
        """
        mask_w, mask_h = mask.size
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

        # obtenemos el formato center a partir del corner
        corner_bbox = [min_x, min_y, width, height]
        bbox = bbox_corn2cent(corner_bbox, mask_w, mask_h)

        return bbox
    

    def _load_from_json(self):
        """
        Cargamos los datos del dataset desde el json
        """
        # cargamos el json
        with open(self.json, "r", encoding="utf-8") as json_file:
            json_dict = json.load(json_file)
        
        # vamos guardando cada dato
        self.resolution_counts = json_dict["resolution_counts"]
        self.light_counts = json_dict["light_counts"]
        self.split_counts = json_dict["split_counts"]
        self.channel_counts = json_dict["channel_counts"]
        self.brightness = json_dict["brightness"]
        self.contrast = json_dict["contrast"]
        self.polyp_centers = json_dict["polyp_centers"]
        self.sum_mask_eucl_dist2center = json_dict["sum_mask_eucl_dist2center"]
        self.mask_heatmap = np.array(json_dict["mask_heatmap"])
        self.sum_masks_percentage = json_dict["sum_masks_percentage"]
        self.dict = json_dict["dict"]

    
    def _save_on_json(self):
        """
        Guardamos los datos en el json con formato:
        {
            datos de analisis del dataset
            diccionario de imágenes{}
        }
        """
        json_dict = {
            "resolution_counts": self.resolution_counts,
            "light_counts": self.light_counts,
            "split_counts": self.split_counts,
            "channel_counts": self.channel_counts,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "polyp_centers": self.polyp_centers,
            "sum_mask_eucl_dist2center": self.sum_mask_eucl_dist2center,
            "mask_heatmap": self.mask_heatmap.tolist(),     # ya que es un np array
            "sum_masks_percentage": self.sum_masks_percentage,
            "dict": self.dict
        }

        with open(self.json, "w", encoding="utf-8") as json_file:
            json.dump(json_dict, json_file)  # `indent=4` para formato legible


class TrainModel:
    """
    Esta clase se encarga del proceso de entrenamiento del modelo dado unos parámetros
    de entrenamiento y unos dataloaders, devolviendo los datos del entrenamiento del modelo
    """

    def __init__(self, model, loss_fn, optim):
                 
        """
        Iniciamos la clase especificando los datos de entrenamiento del modelo.
        model: modelo en pytorch en modo entrenamiento
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        # usamos la GPU mejor si hay libre
        if torch.cuda.is_available():
            gpu_id = self._get_free_gpu()
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"Entrenando en GPU {gpu_id}")
        else:
            self.device = torch.device("cpu")
            print("Entrenando en CPU")

        # cargamos el modelo en el dispositivo
        self.model.to(self.device)

    def train_model(self, num_epoch, train_resolution, 
                    train_dataloader, validation_dataloader, test_dataloader,
                    silent = False):
        """
        Entrenamos el modelo dado con los parámetros especificados
        train_resolution: resolucion a la que transformar las imagenes.
        data_loaders: en formato de ImageDatasetProcessor
        silent: para entrenar el modelo sin prompts
        """
        train_dl = train_dataloader
        test_dl = test_dataloader
        val_dl = validation_dataloader


        # definimos la transofrmacion para el tensor, en 256x256 ya que son patches de 16x16
        transform = transforms.Compose([
            transforms.Resize(train_resolution),
            transforms.ToTensor(),
        ])

        # Estas son variables para analizar el modelo
        log_epochs = 1 # cada cuantas epocas obtenemos datos del modelo
        loss_hist_train = [0] * num_epoch
        loss_hist_val = [0] * num_epoch
        IoU_hist_train = [0] * num_epoch
        IoU_hist_val = [0] * num_epoch

        for epoch in range(num_epoch):  # Número de épocas
            # 📍 Entrenamos el modelo
            model_results = self._try_model(train_dl, self.device, self.model, 
                                            transform, train_mode=True, 
                                            loss_fn=self.loss_fn, 
                                            optimizer=self.optim, 
                                            img_size=train_resolution)
            
            # Guardamos los resultados de la época
            loss_hist_train[epoch], IoU_hist_train[epoch] = model_results

            # 💾 Validamos el modelo
            model_results = self._try_model(val_dl, self.device, self.model, 
                                            transform, loss_fn=self.loss_fn, 
                                            img_size=train_resolution)
            loss_hist_val[epoch], IoU_hist_val[epoch] = model_results

            # mostramos como va el entrenamiento
            if not silent and epoch % log_epochs==0:
                print(f'Epoch {epoch}  Loss train {loss_hist_train[epoch]:.4f}  IoU train {IoU_hist_train[epoch]:.4f} ')
                print(f'Epoch {epoch}  Loss valid {loss_hist_val[epoch]:.4f}  IoU valid {IoU_hist_val[epoch]:.4f} ')

        # 🏁 Finalmente evaluamos el modelo en test
        model_results = self._try_model(test_dl, self.device, self.model, 
                                        transform, loss_fn=self.loss_fn, 
                                        img_size=train_resolution)
        loss_test, IoU_test = model_results

        if not silent:
            self._show_test_results(loss_test, IoU_test)

        # devolvemos los datos para su análisis
        results = { 
            "loss_test": loss_test, 
            "IoU_test": IoU_test,
            "loss_hist_train": loss_hist_train,
            "IoU_hist_train": IoU_hist_train,
            "loss_hist_val": loss_hist_val,
            "IoU_hist_val": IoU_hist_val }

        return results
    
    def show_results(self, dict, save_img=False, img_name="Tmp_res.png"):
        """
        Mostramos los resultados dados como una gráfica, y mostramos el resultado
        final por texto. dado el diccionario results con el formato de "train_model"
        """
        loss_test = dict["loss_test"] 
        IoU_test = dict["IoU_test"]
        loss_hist_train = dict["loss_hist_train"]
        loss_hist_val = dict["loss_hist_val"]
        IoU_hist_train = dict["IoU_hist_train"]
        IoU_hist_val = dict["IoU_hist_val"]

        num_epoch = len(loss_hist_train)

        # Graficar la evolución de la Loss durante el entrenamiento y validación
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)  # Subgráfico 1: Loss
        plt.plot(range(num_epoch), loss_hist_train, label='Loss train', color='blue')
        plt.plot(range(num_epoch), loss_hist_val, label='Loss valid', color='red')
        plt.title('Loss durante el entrenamiento y validación')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()

        # Graficar la evolución de la IoU durante el entrenamiento y validación
        plt.subplot(1, 2, 2)  # Subgráfico 2: IoU
        plt.plot(range(num_epoch), IoU_hist_train, label='IoU train', color='blue')
        plt.plot(range(num_epoch), IoU_hist_val, label='IoU valid', color='red')
        plt.title('IoU durante el entrenamiento y validación')
        plt.xlabel('Épocas')
        plt.ylabel('IoU')
        plt.legend()

        plt.tight_layout()

        # guardamos la imagen si es necesario
        if save_img:
            plt.savefig(img_name, format='png', dpi=300)

        # Mostrar ambas gráficas
        plt.show()


        self._show_test_results(loss_test, IoU_test)

    def _try_model(self, data_loader, device, model, transform, train_mode=False, 
              loss_fn=None, optimizer=None, img_size=(240, 240)):
        """
        Esta funcion se encarga de correr en el modelo el dataloader proporcionado
        aplicando a las imagenes la transformación dada, ejecutando todo en el dispositivo
        indicado y entrenandolo si esta indicado. 
        Si NO se indica entrenar, funciona como una validacion
        """

        # Para seguir el accurracy y el loss del modelo
        loss_try = 0
        IoU_try = 0

        # Escalador para ampliar los gradientes y usar float16 sin perder datos (vanishing de pesos cercanos a 0)
        scaler = torch.cuda.amp.GradScaler()

        for batch in data_loader:
            # Primero debemos cargar las imagen desde su path y convertirlas a tensores
            images = []
            
            # para ello cargamos las imagenes del batch en una lista
            for path in batch['path']:
                image = Image.open(path).convert('RGB') # Aseguramos 3 canales
                # guardamos la imagen transformada
                images.append(transform(image))
            
            # las convertimos en un tensor
            images = torch.stack(images)

            # Ahora procesamos las bbox que parecido a las imagenes vienen como una lista de tensores
            # por lo que las agrupamos y convertimos a un solo tensor
            bbox = torch.stack(batch['bbox']).T

            # Guardamos en GPU
            images = images.to(device, non_blocking=True)   # acelerado el paso a GPU
            bbox = bbox.to(device, non_blocking=True)

            # mixed precision mode, usamos float16 pero reescalamos para evitar perder datos cercanos a 0
            with torch.cuda.amp.autocast():
                pred = model(images)['pred_bboxes']
                loss = loss_fn(bbox, pred)

            if train_mode:
                scaler.scale(loss).backward()       # backward con amplificación
                scaler.step(optimizer)              # actualizamos pesos
                scaler.update()                     # actualizamos escala
                optimizer.zero_grad()

            # Finalmente guardamos el error del batch para analizarlo
            loss_try += loss.item()

            # Obtenemos el valor e IoU del batch
            for pred_box, target_box in zip(pred, bbox):
                IoU_try += yolo_bbox_iou(img_size, pred_box.tolist(), target_box.tolist())

            # 🔻 Limpiamos la VRAM
            del images, bbox, pred, loss
            torch.cuda.empty_cache()

        # Obtenemos la media de error en entrenamiento
        loss_try /= len(data_loader.dataset)
        IoU_try /= len(data_loader.dataset)

        return (loss_try, IoU_try)
    
    def _show_test_results(self, loss_test, IoU_test):
        print("End of training!")
        print("-------------------- FINAL RESULTS ------------------------")
        print(f"|     - Test loss:     {loss_test}                         |")
        print(f"|     - Test IoU: {IoU_test}                         |")
        print("-----------------------------------------------------------")
    
    def _get_free_gpu(self):
        """
        Esta función obtiene la gpu con menor carga de trabajo para evitar errores
        en entrenamiento.
        """
        pynvml.nvmlInit()   # iniciamos ell análisis
        num_devices = pynvml.nvmlDeviceGetCount()   # get the number of GPUs
    
        max_free_mem = 0
        best_gpu = 0

        # Buscamos cuál es la GPU con más VRAM disponible
        for i in range(num_devices):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mem = meminfo.free
            if free_mem > max_free_mem:
                max_free_mem = free_mem
                best_gpu = i
        
        pynvml.nvmlShutdown()   # terminamos el análisis
        return best_gpu

    


############################    Herramientas    ############################

def show_Nresults(list_dict_res, list_dict_names, save_img=False, img_name="Tmp_res.png"):
    """
    Mostramos los resultados de la lista de diccionarios dada, siendo cada diccionario
    el resultado de un benchmark, y mostramos el resultado en una gráfica.
    dado la lista de diccionarios con el formato de "train_model"
    """

    loss_test_mean = 0
    IoU_test_mean = 0
    num_dicts = len(list_dict_res)
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", 
              "gray", "olive", "cyan"]

    plt.figure(figsize=(24, 12))

    # 1️⃣- Loss train
    plt.subplot(2, 2, 1)
    # mostramos cada una de las muestras
    for i, dict in enumerate(list_dict_res):
        loss_hist_train = dict["loss_hist_train"]
        plt.plot(range(len(loss_hist_train)), loss_hist_train, 
                 label=list_dict_names[i], color=colors[i])
    plt.title('Loss Train')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    # 2️⃣- Loss validation
    plt.subplot(2, 2, 3)
    # mostramos cada una de las muestras
    for i, dict in enumerate(list_dict_res):
        loss_hist_val = dict["loss_hist_val"]
        plt.plot(range(len(loss_hist_val)), loss_hist_val, 
                 label=list_dict_names[i], color=colors[i])
    plt.title('Loss Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    # 3️⃣- IoU train
    plt.subplot(2, 2, 2)
    # mostramos cada una de las muestras
    for i, dict in enumerate(list_dict_res):
        IoU_hist_train = dict["IoU_hist_train"]
        plt.plot(range(len(IoU_hist_train)), IoU_hist_train, 
                 label=list_dict_names[i], color=colors[i])
    plt.title('IoU Train')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    # 4️⃣- Loss validation
    plt.subplot(2, 2, 4)
    # mostramos cada una de las muestras
    for i, dict in enumerate(list_dict_res):
        IoU_hist_val = dict["IoU_hist_val"]
        plt.plot(range(len(IoU_hist_val)), IoU_hist_val, 
                 label=list_dict_names[i], color=colors[i])
    plt.title('IoU Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()       

    plt.tight_layout()

    # guardamos la imagen si es necesario
    if save_img:
        plt.savefig(img_name, format='png', dpi=300)

    # Mostrar ambas gráficas
    plt.show()


def load_json_dict(json_path):
    """
    Cargamos los datos del dataset desde el json dado, devolvermos un diccionario
    con los datos del json
    """
    # cargamos el json
    with open(json_path, "r", encoding="utf-8") as json_file:
        json_dict = json.load(json_file)

    return json_dict
    


def bbox_corn2cent(bbox, img_w, img_h):
    """"
    Esta funcion procesa las bboxes del formato "corner" usado en COCO [x, y, w, h] (0-num pixels)
    a formato "center" usado en la salida de YOLO [cx, cy, w, h] normalizado (0-1)
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


def bbox_cent2corn(cent_bbox, img_w, img_h):
    """
    Esta funcion pasa del formato "center" usado en YOLO noramalizado de [cx, cy, w, h]
    al formato "corner" usado en COCO en coordenadas de la imagen [x, y, w, h] -> xy esq sup izq
    """
    cx_cent, cy_cent, w_cent, h_cent= cent_bbox


    w = int(w_cent * img_w)
    h = int(h_cent * img_h)
    cx = cx_cent * img_w
    cy = cy_cent * img_h
    x = cx - (w / 2)
    y = cy - (h / 2)

    # Hacemos los números enteros para evitar fallos
    return [int(x), int(y), int(w), int(h)]

def bbox_corn2doublecorn(corn_bbox):
    """
    Esta funcion pasa del formato "corner" usado en COCO [x, y, w, h] -> xy esq sup izq
    al formato "double corner" usado en IoU de PyTorch [minx, miny, maxx, maxy]
    """
    x_min, y_min, w, h= corn_bbox
    
    x_max = x_min + w
    y_max = y_min + h

    # Hacemos los números enteros para evitar fallos
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def bbox_cent2doublecorn(cent_bbox, img_w, img_h):
    """
    Esta funcion pasa del formato "center" usado en YOLO noramalizado de [cx, cy, w, h]
    al formato "double corner" usado en IoU de PyTorch [minx, miny, maxx, maxy]
    """
    # Primero pasamos ambas bbox de formato yolo (cx, cy, w, h) normalizado
    # al formato de bbox de IoU (xmin, ymin, xmax, ymax)
    dcorn_bbox = bbox_corn2doublecorn(bbox_cent2corn(cent_bbox, img_w, img_h))

    # restringimos los valores de pred_dcorn por si son negativos
    dcorn_bbox = [max(0, dcorn_bbox[0]), max(0, dcorn_bbox[1]), 
                  max(0, dcorn_bbox[2]), max(0, dcorn_bbox[3])]

    return dcorn_bbox

def yolo_bbox_iou(img_size, targ_box, pred_box):
    """
    Obtiene el inidce IoU de coincidencia de las bbox dadas
    """
    # Primero transformamos las bboxes para IoU "double corner" (xmin,ymin,xmax,ymax)
    pred_dcorn = bbox_cent2doublecorn(pred_box, img_size[0], img_size[1])
    targ_dcorn = bbox_cent2doublecorn(targ_box, img_size[0], img_size[1])

    pred_t = torch.tensor(pred_dcorn).unsqueeze(0)  # guardamos todas las bboxes en un tensor
    targ_t = torch.tensor(targ_dcorn).unsqueeze(0)

    # calculamos el IoU de las bbox
    iou_res = torch.nan_to_num(box_iou(pred_t, targ_t), nan=0.0)# evitamos nan si es 0
    return torch.sum(iou_res).item()
