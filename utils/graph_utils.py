"""
En este documento se encapsulan las funcionalidades de muestra y graficar datos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches 
import seaborn as sns
from PIL import Image
from scipy.stats import gaussian_kde
from torchvision.ops import box_convert
import utils


#################################  TEXTO  ######################################

def show_test_results(loss_test, IoU_test):
    print("End of training!")
    print("-------------------- FINAL RESULTS ------------------------")
    print(f"|     - Test loss:\t{loss_test:-3f}                      |")
    print(f"|     - Test IoU:\t{IoU_test:.3f}                        |")
    print("-----------------------------------------------------------")


def print_summary(idp):
    """
    Imprime un resumen de las estadísticas del dataset.
    """
    print(f"Total imágenes: {len(idp.polyp_centers)}")

    print("Composición del dataset:")
    for dictionary in [
            idp.resolution_counts,
            idp.light_counts, 
            idp.split_counts, 
            idp.channel_counts,
            idp.paris_count]:
        if dictionary == idp.resolution_counts:
            print(f"Resoluciónes: total distintas resoluciones {len(idp.resolution_counts)}")
        elif dictionary == idp.light_counts:
            print("Tipos de luz:")
        elif dictionary == idp.split_counts:
            print("Splits:")
        elif dictionary == idp.paris_count:
            print("Tipos de lesión:")
        else:
            print("Canales:")
        for data, num in dictionary.items():
            print(f"\t{data}: {num}", end="")
        print("\n")

    mean_masks_percentage = idp.sum_masks_percentage / len(idp.polyp_centers)
    print(f"Volumen medio de los pólipos respecto a la imagen:\t{mean_masks_percentage}%")
    mean_bbox_percentage = idp.sum_bbox_percentage / len(idp.polyp_centers)
    print(f"Volumen medio de las bbox respecto a la imagen:\t{mean_bbox_percentage}%")
    mean_mask_eucl_dist2center = idp.sum_mask_eucl_dist2center / len(idp.polyp_centers)
    print(f"Distancia media del centro del pólipos al centro de la imagen:\t{mean_mask_eucl_dist2center}px")



#################################  IMÁGEN  #####################################


def show_results(dict, save_img=False, img_name="Tmp_res.png", eval_pred=False):
    """
    Mostramos los resultados dados como una gráfica, y mostramos el resultado
    final por texto. dado el diccionario results con el formato de "train_model"
    Graficamos también los datos de evaliació si son requeridos
    """
    colors = ["red", "blue", "darkgreen", "darkviolet", 
              "gold", "black", "saddlebrown", "teal"]

    loss_test = dict["loss_test"] 
    IoU_test = dict["IoU_test"]
    loss_hist_train = dict["loss_hist_train"]
    loss_hist_val = dict["loss_hist_val"]
    IoU_hist_train = dict["IoU_hist_train"]
    IoU_hist_val = dict["IoU_hist_val"]
    eval_data = dict["eval_data"]

    num_epoch = len(loss_hist_train)

    # 📈 Primero graficamos los resultados del entrenamiento
    plt.figure(figsize=(14, 8))
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

    # mostramos y guardamos la imágen
    plt.tight_layout()
    if save_img:
        plt.savefig(img_name, format='png', dpi=300)
    plt.show()

    # 🧪 Segundo graficamos los datos de evaliación si se requieren
    # medias de volumen en imagen de la bbox [train, validación, test]
    pred_box_vol_means = []     # predichas
    true_box_vol_means = []     # verdaderas
    # centro de las bbox [train, validación, test]
    pred_bbox_centers = [[], [], []]
    true_bbox_centers = [[], [], []]
    pred_sum_areas = 0
    true_sum_areas = 0

    if eval_pred:
        # para cada split guardamos los centros de las bbox
        for i, split in enumerate(eval_data):
            for sample in split:
                # predichas
                cx, cy, w, h = utils.bbox_doublecorn2center(sample["pred_bbox"])
                pred_bbox_centers[i].append((cx, cy))
                pred_sum_areas += w*h
                # verdaderas
                cx, cy, w, h = utils.bbox_doublecorn2center(sample["true_bbox"])
                true_bbox_centers[i].append((cx, cy))
                true_sum_areas += w*h
            pred_box_vol_means.append(pred_sum_areas/len(split))
            true_box_vol_means.append(true_sum_areas/len(split))
            true_sum_areas = 0
            pred_sum_areas = 0

        # 1️⃣ - Área de ocuipación de la bbox verdadera en la imágen
        x = np.arange(len(true_box_vol_means))
        bar_width = 0.6
        split_names = ["train", "validation", "test"]

        plt.figure(figsize=(12, 12))
        plt.subplot(4, 2, 1)  
        plt.bar(x, true_box_vol_means, width=bar_width, 
                color=colors[:len(eval_data)])
        plt.title('Área de bbox verdadera en la imágen')
        plt.xticks(x, split_names)
        for i, value in enumerate(true_box_vol_means):
            plt.text(x[i], value + 0.01, f"{value:.2f}%", ha='center', va='bottom', fontsize=9)

        # 2️⃣ - Área de ocuipación de la bbox predicha en la imágen
        plt.subplot(4, 2, 2)  
        plt.bar(x, pred_box_vol_means, width=bar_width, 
                color=colors[:len(eval_data)])
        plt.title('Área de bbox predicha en la imágen')
        plt.xticks(x, split_names)
        for i, value in enumerate(pred_box_vol_means):
            plt.text(x[i], value + 0.01, f"{value:.2f}%", ha='center', va='bottom', fontsize=9)

        # 3️⃣ - Centros de las bbox predichas y objetivo de cada split
        for i in range(len(pred_bbox_centers)):
            plt.subplot(4, 2, 3+(i*2))
            set_heat_point_map(plt.gca(), plt.gcf(), 
                            f"Centros de bboxes verdaderas {split_names[i]}", 
                            true_bbox_centers[i], (1, 1))
            plt.subplot(4, 2, 4+(i*2))
            set_heat_point_map(plt.gca(), plt.gcf(), 
                            f"Centros de bboxes predichas {split_names[i]}", 
                            pred_bbox_centers[i], (1, 1))

        plt.tight_layout()
        
        # guardamos la imagen si es necesario
        if save_img:
            plt.savefig(img_name.replace(".png", "_data.png"), format='png', dpi=300)

        # Mostrar ambas gráficas
        plt.show()

    show_test_results(loss_test, IoU_test)




def graph_summary(idp):
    # Configuración del estilo de los gráficos
    sns.set(style="whitegrid")

    colors = plt.cm.tab10.colors  # Colores por defecto


    # Crear ventana con gráficos
    fig, axs = plt.subplots(2, 4, figsize=(19, 6))

    # Graficamos los diagramas
    charts = [
        # Gráfico 1: Distribución de las imágenes por split
        (idp.split_counts, axs[0, 0], 'División de imágenes del dataset', 'Número de Imágenes'),
        # Gráfico 2: Composición del dataset por tipo de luz
        (idp.light_counts, axs[0, 1], 'Composición del dataset por tipo de luz', '% del dataset'),
        # Gráfico 3: Tipos de resoluciones en las imágenes del dataset
        (idp.resolution_counts, axs[0, 2], 'Tipos de resoluciones en las imágenes del dataset', '% del dataset'),
        # Gráfico 4: Número de canales por tipo de imágen
        (idp.channel_counts, axs[0, 3], 'Formato de las imágenes', '% del dataset')
    ]

    for data, ax, title, ylabel in charts:
        ax.set_ylabel(ylabel)
        ax.bar(data.keys(), data.values(), color=colors[:len(data.keys())])
        ax.set_title(title)

    # Graficamos los histogramas
    hist = [
        # Gráfico 5: Histograma del brillo en las imágenes
        (idp.brightness, axs[1, 0], 'Brillo de los frames', 'Número de Imágenes'),
        # Gráfico 6: Histograma del contraste en las imágenes
        (idp.contrast, axs[1, 1], 'Contraste de los frames', 'Número de Imágenes')
    ]
    
    for data, ax, title, ylabel in hist:
        ax.hist(data, bins=20, color='forestgreen')
        ax.set_title(title)
        ax.set_ylabel(ylabel)


    # Gráfico 7: heatmap de distribución de las máscaras
    sns.heatmap(idp.mask_heatmap, cmap="crest", ax=axs[1, 2], cbar=True,
                xticklabels=False, yticklabels=False)
    axs[1, 2].set_title('Distribución de las máscaras (Heatmap)')
    axs[1, 2].set_box_aspect(idp.target_resolution[1] / idp.target_resolution[0])

    # Gráfico 8: muestra de los centros de los pólipos
    cx, cy = zip(*idp.polyp_centers)       # dos listas con coordenadas
    cx = np.array(cx)
    cy = np.array(cy)

    # Crear un mapa de densidad 2D
    xy = np.vstack([cx, cy])
    kde = gaussian_kde(xy)
    z = kde(xy)

    # Ordenar por densidad para que los puntos más densos se vean encima
    idx = z.argsort()
    cx, cy, z = cx[idx], cy[idx], z[idx]

    # Mostrar puntos con color según densidad
    sc = axs[1, 3].scatter(cx, cy, c=z, s=5, cmap='YlOrRd', alpha=0.8)
    axs[1, 3].set_title('Densidad de centros de pólipos')
    axs[1, 3].set_xlim([0, idp.target_resolution[0]])
    axs[1, 3].set_ylim([0, idp.target_resolution[1]])
    axs[1, 3].set_box_aspect(idp.target_resolution[1] / idp.target_resolution[0])

    # Colorbar
    cbar = fig.colorbar(sc, ax=axs[1, 3])
    cbar.set_label('Densidad estimada')


    # Ajustar el layout
    plt.tight_layout()
    plt.show()

    # Gráfico 9: muesta los tipos de lesión por clasificación de parís
    if idp.meta_path is not None:
        data_dict = idp.paris_count
        total_img = sum(data_dict.values())
        labels = list(data_dict.keys())

        # calculamos los porcentajes de cada lesión
        percentages = [(v / total_img * 100) if total_img > 0 else 0 for v in data_dict.values()]
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(labels, percentages, color=colors[:len(data_dict)])
        ax.set_ylabel('Porcentaje de Imágenes (%)')
        ax.set_title("Volumen de imágenes por tipo de lesión")
        set_bar_percentage_format(ax, bars)

        # Rotamos las etiquetas del eje X para mejor legibilidad
        set_bar_percentage_format(ax, bars)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)

        ax.set_ylim(0, max(percentages) + 10)  # Espacio para etiquetas


        plt.tight_layout()
        plt.show()


def graph_Nsummarys(list_idps):
    """
    mostramos en conjunto los datos de los datasets dados para comparar
    - list_idps: Lista con los ipds cuyos datos se muestran juntos
    """
    # Configuración del estilo de los gráficos
    sns.set(style="whitegrid")

    common_rows = 4 # columnas con grafiocs comunes
    idp_names = []

    # Crear ventana con gráficos
    fig, axs = plt.subplots(common_rows+(len(list_idps)), 2, figsize=(16, 24))

    # Estructura de los graficos de barras
    charts = [
        # Gráfico 1: Distribución de las imágenes por split
        [[], axs[0, 0], 'División de imágenes del dataset', 'Número de Imágenes'],
        # Gráfico composición del dataset por tipo de luz
        [[], axs[0, 1], 'Composición del dataset por tipo de luz', 'Número de Imágenes'],
        # Gráfico tipos de resoluciones en las imágenes del dataset
        [[], axs[1, 0], 'Tipos de resoluciones en las imágenes del dataset', 'Número de Imágenes'],
        # Gráfico número de canales por tipo de imágen
        [[], axs[1, 1], 'Formato de las imágenes', 'Número de Imágenes'],
        # Gráfico ocupación de la máscara en pantalla
        [[], axs[2, 0], 'Volumen de la máscara', 'Porcentaje de ocupación'],
        # Gráfico ocupación de la bbox en pantalla
        [[], axs[2, 1], 'Volúmen de la bbox', 'Porcentaje de ocupación']
    ]

    # Estructura de los histogramas
    hist = [
        # Gráfico 5: Histograma del brillo en las imágenes
        ([], axs[3, 0], 'Brillo de los frames', 'Número de Imágenes'),
        # Gráfico 6: Histograma del contraste en las imágenes
        ([], axs[3, 1], 'Contraste de los frames', 'Número de Imágenes')
    ]

    # obtenemos los datos a graficar
    for idp in list_idps:
        idp_names.append(idp.name)
        charts[0][0].append(idp.split_counts)
        charts[1][0].append(idp.light_counts)
        charts[2][0].append(idp.resolution_counts)
        charts[3][0].append(idp.channel_counts)
        charts[4][0].append(idp.sum_masks_percentage/len(idp.polyp_centers))
        charts[5][0].append(idp.sum_bbox_percentage/len(idp.polyp_centers))
        hist[0][0].append(idp.brightness)
        hist[1][0].append(idp.contrast)
    

    # 📊 graficos de barras 
    for data, ax, title, ylabel in charts:
        if title is 'División de imágenes del dataset':
            set_bar_graph(data, ax, title, ylabel, idp_names, percentages=False)
        else:
            set_bar_graph(data, ax, title, ylabel, idp_names)
    # 📈 histogramas
    for data, ax, title, ylabel in hist:
        set_histogram(data, ax, title, ylabel, idp_names)

    # ⚗️ heatmaps y localización de los centros
    for i, idp in enumerate(list_idps):
        row=common_rows+i
        # Gráfico 7: heatmap de distribución de las máscaras
        sns.heatmap(idp.mask_heatmap, cmap="crest", ax=axs[row, 0], cbar=True,
                    xticklabels=False, yticklabels=False)
        axs[row, 0].set_title(f'Distribución de las máscaras {idp_names[i]}')
        axs[row, 0].set_box_aspect(idp.target_resolution[1] / idp.target_resolution[0])

        # Gráfico 8: muestra de los centros de los pólipos
        set_heat_point_map(axs[row, 1], fig, f"Centros de los pólipos en {idp_names[i]}",
                           idp.polyp_centers, idp.target_resolution)


    # Ajustar el layout
    plt.tight_layout()
    plt.show()

    # mostramos el gráfico con cada tipo de lesión
    data_dicts = []
    for idp in list_idps:
        if idp.meta_path is not None:
            data_dicts.append(idp.paris_count)
    if len(data_dicts) != 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        set_bar_graph(data_dicts, ax, 
                        "Volumen de imágenes por tipo de lesión", 
                        'Número de Imágenes', idp_names)

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.show()



def set_bar_graph(data_list, ax, title, ylabel, set_names, percentages=True):
    """
    Define un grafico de barras dentro de un mosaico de gráficos
    - percentages: hace set del gráfico como porcentajes
    """

    colors = ["red", "blue", "darkgreen", "darkviolet", 
              "gold", "black", "saddlebrown", "teal"]
    ax.set_ylabel(ylabel)  # Configura el label del eje Y
    ax.set_title(title)    # Configura el título del gráfico

    bar_width = 0.2  # Ancho de las barras

    # con varios datos graficamso juntos
    if isinstance(data_list[0], dict):
        # obtenemos todas los tipos de datos a graficar
        all_keys = sorted(set().union(*[d.keys() for d in data_list])) 
        x_pos = np.arange(len(all_keys))  # Posiciones en el eje X para cada clave

        for i, d in enumerate(data_list):
            total_imgs = sum(d.values())    # obtenemos el total de imágenes
            # obtiene el dato en % para esa key, si no hay es 0
            if percentages:
                values = [d.get(k, 0) / total_imgs * 100 if total_imgs > 0 else 0 for k in all_keys]
            else:
                values = [d.get(k, 0) for k in all_keys]    # obtiene el dato para esa key, si no hay es 0
           
            # Desplazamos las barras para que no se superpongan
            bars = ax.bar(x_pos + i * bar_width, values, bar_width, 
                    label=set_names[i])
            # ponemos el porcentaje de la barra
            if percentages:
                set_bar_percentage_format(ax, bars)

        ax.set_xticks(x_pos + bar_width * (len(data_list) / 2 - 0.5))  # Centra las etiquetas
        ax.set_xticklabels(all_keys)  # Etiquetas para las barras
        ax.legend()
    else:   # para un dato un valor por barra
        total = sum(data_list)
        values = data_list
        if percentages:
            values = [(v / total * 100) if total > 0 else 0 for v in data_list]

        x_pos = np.arange(len(data_list))

        bars = ax.bar(x_pos, values, color=colors[:len(data_list)], tick_label=set_names)

        if percentages:
            set_bar_percentage_format(ax, bars)
        
    if percentages:
        ax.set_ylim(0, 110)  # Asegura espacio para las etiquetas arriba de 100%


def set_bar_percentage_format(ax, bars):
    """
    Función para incluir texto de porcentaje encima de las barras del grafo
    """

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)



def set_histogram(data_list, ax, title, ylabel, set_names):
    """
    Define un histograma dentro de un mosaico de gráficos
    """
    colors = ["red", "blue", "darkgreen", "darkviolet", 
              "gold", "black", "saddlebrown", "teal"]

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.hist(data_list, bins=20, color=colors[:len(set_names)], 
            alpha=0.5, label=set_names)
    ax.legend()


def set_heat_point_map(ax, fig, name, list_centers, target_resolution):
    """
    Define un grafico de puntos indicando con colores la densidad, 
    dentro de un mosaico de gráficos
    """
    cx, cy = zip(*list_centers)       # dos listas con coordenadas
    cx = np.array(cx)
    cy = np.array(cy)

    # Crear un mapa de densidad 2D
    xy = np.vstack([cx, cy])
    kde = gaussian_kde(xy)
    z = kde(xy)

    # Ordenar por densidad para que los puntos más densos se vean encima
    idx = z.argsort()
    cx, cy, z = cx[idx], cy[idx], z[idx]

    # Mostrar puntos con color según densidad
    sc = ax.scatter(cx, cy, c=z, s=5, cmap='YlOrRd', alpha=0.8)
    ax.set_title(name)
    ax.set_xlim([0, target_resolution[0]])
    ax.set_ylim([0, target_resolution[1]])
    ax.set_box_aspect(target_resolution[1] / target_resolution[0])

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Densidad estimada')


def show_Nresults(list_dict_res, list_dict_names, save_img=False, img_name="Tmp_res.png"):
    """
    Mostramos los resultados de la lista de diccionarios dada, siendo cada diccionario
    el resultado de un benchmark, y mostramos el resultado en una gráfica.
    dado la lista de diccionarios con el formato de "train_model"
    """

    loss_test_mean = 0
    IoU_test_mean = 0
    num_dicts = len(list_dict_res)
    colors = [
        "red", "darkorange",
        "blue", "dodgerblue",
        "darkgreen", "limegreen",
        "darkviolet", "deeppink",      
        "gold", "yellow",
        "black", "silver",
        "saddlebrown", "chocolate",
        "teal", "darkturquoise"
    ]

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

    
def show_image(path, tar_bbox, pred_bbox=None, name=None, ax=None, paris_class=None):
    """
    Esta funcion muestra la imagen con la bbox objetivo asociada y la predicha si dada
    Mostramos la imagen, imprimimos su ID y graficamos las bbox dadas, también 
    se enmarca dentro de un subplot si se da un ax
    - path: path completo a la imágen a mostrar
    - tar_bbox: bbox objetivo a mostrar formato YOLO o "center", (xc, cy, w, h)
    - pred_bbox: bbox predicha a mostrar formato YOLO o "center", (xc, cy, w, h)
    - name: nombre para mostrar como título
    - ax: posición en un gráfico con más imágenes si dado
    - paris_class: clasificación de paris de la lesión
    """
    img = mpimg.imread(path)
    img_h, img_w = img.shape[:2]
    object_bbox = utils.bbox_doublecorn2corn(tar_bbox, img_w, img_h, format="real")

    if pred_bbox is not None:
        pred_bbox = utils.bbox_doublecorn2corn(pred_bbox, img_w, img_h, format="real")
    
    # Creamos la figura para añadir los datos si no es dada ya
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 10))

    # Cargamos la imagen en la figura
    ax.imshow(img)

    # Añadimos el dibujo de la bbox objetivo
    x, y, w, h = object_bbox
    rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                            edgecolor='cyan', facecolor="none", 
                            label="Objetivo") 
    ax.add_patch(rect)
    
    # Añadimos el dibujo de la bbox predicha si dada
    if pred_bbox is not None:
        x, y, w, h = pred_bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                edgecolor='red', facecolor="none",
                                label="Predicha") 
        ax.add_patch(rect)
    
    if name is not None:
        if paris_class is not None:
            name = f"{name}\n{paris_class}"
        ax.set_title(name, fontsize=16, loc='center')
    
    # Mostrar la leyenda
    ax.legend(loc='upper left', fontsize=12)
            
    # Si 'ax' no fue pasado, mostramos la imagen
    if ax is None:
        plt.show()
