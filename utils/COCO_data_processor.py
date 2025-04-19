"""
    Esta es una clase para procesar el dataset de COCO para el entranemaiento del modelo de YOLO
"""


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