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
import gc
import json
import csv
import json
import torch
import pynvml
import graph_utils
import numpy as np
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
        - json_path: path al json del dataset, si no existe se crea. Si es None no se guarda
        - polyp_paths: path a las imagenes de polipos, uno o varios directorios
        - mask_paths: path a las imagenes de macaras, uno o varios directorios
        - void_paths: path a las imagenes de void , uno o varios directorios
        """

        # contexto que nos da el usuario
        self.target_resolution = target_resolution      # ancho x alto
        self.json = json_path
        self.name = dataset_name

        # Información sobre el dataset
        self.resolution_counts = {}     # Conteo de imágenes por resolución
        self.light_counts = {}          # Número de imágenes por formato: WLI, NBI ...
        self.split_counts = {}          # Número de imágenes por funcion; test, train ...
        self.channel_counts = {}        # Conteo de imágenes por número de canales
        self.brightness = []            # Brillo de las imagenes
        self.contrast = []              # Contraste de las imágenes
        self.polyp_centers = []         # Centros de los pólipos
        # suma de la dist euclidea del centro del ṕolipo respecto al centro img
        self.sum_mask_eucl_dist2center = 0
        # heatmap con la distribución de las máscaras
        self.mask_heatmap = np.zeros(target_resolution, dtype=np.float32)
        # suma de porcentage de ocupación de las máscaras en la imagen
        self.sum_masks_percentage = 0
        # suma de porcentage de ocupación de las bbox en la imagen
        self.sum_bbox_percentage = 0


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
        if self.json != None and os.path.isfile(self.json):
            self._load_from_json()

    
    def show_image(self, image_id):
        graph_utils.show_image(self.dict[image_id]["path"], 
                               self.dict[image_id]["bbox"])

    def print_summary(self):
        graph_utils.print_summary(self)

    def graph_summary(self):
        graph_utils.graph_summary(self)


    def load_dataset(self, polyps_path=None, masks_path=None, voids_path=None,
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
            if bbox == (0, 0, 0, 0):    # si está vacío lo decimos
                print(f"Imagen {img_name} vacía!")

            # actualizamos las estadísticas del dataset
            self._update_stats(img, mask, void_path, bbox, light_type, split, img_name)
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
        
        # finalmente guardamos los nuevos datos en el diccionario
        self.dict.update(image_dict)

        # y sobreescribimos el json con la nueva información si existe
        if self.json is not None:
            self._save_on_json()

    
    def get_dataloaders(self, batch_size, use_premade_splits=False, 
                        analize_splits=True, rand=False,
                        train_split=0, val_split=0, test_split=0):
        """
        Función que divide el diccionario del dataset los tres conjuntos de train
        validation y test.
        
        - use_premade_splits: puedes usar el conjunto prehecho del datset si lo hay
        o indicar el porcentaje de cada split.
        - analize_splits: analiza la imágenes de cada split y muestra sus características
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

        # mostramos información sobre la composición de los splits
        if analize_splits:
            self._evaluate_splits(train_ids, val_ids, test_ids)


        # Finalmente creamos los dataLoaders de las imágenes de cada split
        # suffle para mezclar los datos en cada época 
        # pin_memory para acelerar CPU->GPU
        train_loader = DataLoader(dtrain, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(dval, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(dtest, batch_size=batch_size, shuffle=True, pin_memory=True)

        return train_loader, val_loader, test_loader
    

    def print_summary(self):
        graph_utils.print_summary(self)

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


    def _update_stats(self, img, mask, void_path, bbox, light_type, split, img_name):
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

        # 7️⃣- Porcentaje de ocupación de la bbox en pantalla
        _, _, box_w, box_h = bbox       # (cx, cy, w, h)
        percentage = (box_w * box_h) * 100
        self.sum_bbox_percentage += percentage

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


    def _evaluate_splits(self, train_ids, val_ids, test_ids):
        """
        Devuelve un análisis detallado de cada split.
        Usamos para ello los ids del diccionario del datataset completo
        """        

        # Creamos un idp para cada split
        train_idp = self._get_split_processor(train_ids, split_name="train")
        val_idp = self._get_split_processor(val_ids, split_name="validation")
        test_idp = self._get_split_processor(test_ids, split_name="test")


        print("\n\nResumen TRAIN:")
        train_idp.print_summary()
        print("\n\nResumen VALIDATION:")
        val_idp.print_summary()
        print("\n\nResumen TEST:")
        test_idp.print_summary()

        graph_utils.graph_Nsummarys([train_idp, val_idp, test_idp])


    def _get_split_processor(self, ids, split_name="split"):
        """
        Crea un nuevo ImageDatasetProcessor "fantasma" con el subconjunto de imágenes dado,
        este solo contiene datos de análisis sin duplicar diccionarios
        - ids: ids del diccionario de self con las imágenes del subconjunto
        - name: nomrbe dado al nuevo IDP
        """
        # formato del nombre del path al json
        json_path = self.json.replace(".json", f"_{split_name}_evaluation.json")

        # un idp vacío para evitar duplicar datos
        split_idp = ImageDatasetProcessor(target_resolution=self.target_resolution, 
                                          dataset_name=split_name, 
                                          json_path=json_path)

        # si ya lo hemos procesado lo cargamos del json
        if self.json != None and os.path.isfile(json_path):
            split_idp._load_from_json()
            return split_idp

        # si no, actualizamos la información del split por cada imágen
        for id in ids:
            data = self.dict[id]    # usamos el dato del dataset    
            img = Image.open(data["path"])
            mask = Image.open(data["mask_path"])

            split_idp._update_stats(img, mask, data["void_path"], data["bbox"],
                                    data["light_type"], split_name, id)
            img.close()
            mask.close()

            # guardamos los datos en un json
            if split_idp.json != None:
                split_idp._save_on_json()
        
        # Ajustamos las medias
        split_idp.sum_masks_percentage /= len(split_idp.polyp_centers)
        split_idp.sum_bbox_percentage /= len(split_idp.polyp_centers)
        split_idp.sum_mask_eucl_dist2center /= len(split_idp.polyp_centers)
        
        return split_idp


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
        self.sum_bbox_percentage = json_dict["sum_bbox_percentage"]
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
            "sum_bbox_percentage": self.sum_bbox_percentage,
            "dict": self.dict
        }

        with open(self.json, "w", encoding="utf-8") as json_file:
            json.dump(json_dict, json_file)  # `indent=4` para formato legible


class TrainModel:
    """
    Esta clase se encarga del proceso de entrenamiento del modelo dado unos parámetros
    de entrenamiento y unos dataloaders, devolviendo los datos del entrenamiento del modelo
    """

    def __init__(self, model, loss_fn, optim, eval_pred=False):
                 
        """
        Iniciamos la clase especificando los datos de entrenamiento del modelo.
        - model: modelo en pytorch en modo entrenamiento
        - eval_pred: modo de entrenamiento para evaluar los resultados, guarda
        datos de la úlitma époch
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.eval_pred = eval_pred
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
        - train_resolution: resolucion a la que transformar las imagenes.
        - data_loaders: en formato de ImageDatasetProcessor
        - silent: para entrenar el modelo sin prompts
        """
        train_dl = train_dataloader
        test_dl = test_dataloader
        val_dl = validation_dataloader
        eval_try = False                # recogemos datos extra de entrenamiento para evaluar
        eval_data = None

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
            # Analizamos la última époch si necesitamos datos de análisis
            if self.eval_pred and epoch == num_epoch-1:
                eval_try = True
            # 📍 Entrenamos el modelo
            model_results = self._try_model(train_dl, self.device, self.model, 
                                            transform, train_mode=True, 
                                            loss_fn=self.loss_fn, 
                                            optimizer=self.optim, 
                                            img_size=train_resolution,
                                            eval_try=eval_try)
            
            # Guardamos los resultados de la época
            loss_hist_train[epoch], IoU_hist_train[epoch], train_data = model_results

            # 💾 Validamos el modelo
            model_results = self._try_model(val_dl, self.device, self.model, 
                                            transform, loss_fn=self.loss_fn, 
                                            img_size=train_resolution, 
                                            eval_try=eval_try)
            loss_hist_val[epoch], IoU_hist_val[epoch], val_data = model_results

            # mostramos como va el entrenamiento
            if not silent and epoch % log_epochs==0:
                print(f'Epoch {epoch}  Loss train {loss_hist_train[epoch]:.4f}  IoU train {IoU_hist_train[epoch]:.4f} ')
                print(f'Epoch {epoch}  Loss valid {loss_hist_val[epoch]:.4f}  IoU valid {IoU_hist_val[epoch]:.4f} ')

        # 🏁 Finalmente evaluamos el modelo en test
        model_results = self._try_model(test_dl, self.device, self.model, 
                                        transform, loss_fn=self.loss_fn, 
                                        img_size=train_resolution,
                                        eval_try=self.eval_pred)
        loss_test, IoU_test, test_data = model_results

        if not silent:
            graph_utils.show_test_results(loss_test, IoU_test)
        
        # guardamos los datos de análisis para mostrarlos
        if self.eval_pred:
            eval_data = [train_data, val_data, test_data]

        # devolvemos los datos para su análisis
        results = { 
            "loss_test": loss_test, 
            "IoU_test": IoU_test,
            "loss_hist_train": loss_hist_train,
            "IoU_hist_train": IoU_hist_train,
            "loss_hist_val": loss_hist_val,
            "IoU_hist_val": IoU_hist_val,
            "eval_data": eval_data}

        return results
    

    def show_results(self, dict, save_img=False, img_name="Tmp_res.png"):
        graph_utils.show_results(dict, save_img=save_img, img_name=img_name, 
                                 eval_pred=self.eval_pred)


    def _try_model(self, data_loader, device, model, transform, train_mode=False, 
              loss_fn=None, optimizer=None, img_size=(240, 240), eval_try=False):
        """
        Esta funcion se encarga de correr en el modelo el dataloader proporcionado
        aplicando a las imagenes la transformación dada, ejecutando todo en el dispositivo
        indicado y entrenandolo si esta indicado. 
        Si NO se indica entrenar, funciona como una validacion
        - eval_try: guardamos las predicciones del try para su evaluación
        """
        # Aligeramos la carga en memoria
        gc.collect()
        torch.cuda.empty_cache()


        # Para seguir el accurracy y el loss del modelo
        loss_try = 0
        IoU_try = 0

        total_samples = 0

        # guardar la salida del modelo para evaluar
        eval_data = []

        # Escalador para ampliar los gradientes y usar float16 sin perder datos (vanishing de pesos cercanos a 0)
        scaler = torch.amp.GradScaler()

        for batch in data_loader:
            # Primero debemos cargar las imagen desde su path y convertirlas a tensores
            images = []

            # obtener el número exacto de datos del try
            batch_size = len(batch['path'])
            total_samples += batch_size
            
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
            with torch.amp.autocast(device_type=self.device.type):
                pred = model(images)['pred_bboxes']
                loss = loss_fn(bbox, pred)

            if train_mode:
                scaler.scale(loss).backward()       # backward con amplificación
                scaler.step(optimizer)              # actualizamos pesos
                scaler.update()                     # actualizamos escala
                optimizer.zero_grad()

            # Finalmente guardamos el error del batch para analizarlo
            loss_try += loss.item()

            # Obtenemos el valor e IoU del batch y guardamos datos de evaluación
            for i, (pred_box, target_box) in enumerate(zip(pred, bbox)):
                IoU_img = yolo_bbox_iou(img_size, pred_box.tolist(), target_box.tolist())
                IoU_try += IoU_img
                
                # guardamos dátos de anaĺisis si es requerido
                if eval_try:
                    eval_data.append({
                        "image_path": batch['path'][i],
                        # guardamos los datos fuera de la GPU
                        "pred_bbox": pred_box.detach().cpu().tolist(),
                        "true_bbox": target_box.detach().cpu().tolist(),
                        "IoU": IoU_img
                    })
                    
            # 🔻 Limpiamos la VRAM
            del images, bbox, pred, loss
            torch.cuda.empty_cache()

        # Obtenemos la media de error en entrenamiento
        loss_try /= total_samples
        IoU_try /= total_samples

        return (loss_try, IoU_try, eval_data)
    
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
