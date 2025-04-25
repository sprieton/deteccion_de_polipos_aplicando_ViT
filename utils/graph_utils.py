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
            idp.channel_counts]:
        if dictionary == idp.resolution_counts:
            print(f"Resoluciónes: total distintas resoluciones {len(idp.resolution_counts)}")
        elif dictionary == idp.light_counts:
            print("Tipos de luz:")
        elif dictionary == idp.split_counts:
            print("Splits:")
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


def show_results(dict, save_img=False, img_name="Tmp_res.png"):
    """
    Mostramos los resultados dados como una gráfica, y mostramos el resultado
    final por texto. dado el diccionario results con el formato de "train_model"
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

    # obtenemos datos a graficar
    box_vol_means = []              # medias de ocupación de las bbox en train, val y test
    bbox_centers = [[], [], []]     # centros de las bbox respecto a la imágen
    sum_areas = 0
    for i, split in enumerate(eval_data):
        for cx, cy, w, h in split:
            sum_areas += w*h
            bbox_centers[i].append((cx, cy))
        box_vol_means.append(sum_areas/len(split))


    # Graficar la evolución de la Loss durante el entrenamiento y validación
    plt.figure(figsize=(14, 8))
    plt.subplot(3, 2, 1)  # Subgráfico 1: Loss
    plt.plot(range(num_epoch), loss_hist_train, label='Loss train', color='blue')
    plt.plot(range(num_epoch), loss_hist_val, label='Loss valid', color='red')
    plt.title('Loss durante el entrenamiento y validación')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    # Graficar la evolución de la IoU durante el entrenamiento y validación
    plt.subplot(3, 2, 2)  # Subgráfico 2: IoU
    plt.plot(range(num_epoch), IoU_hist_train, label='IoU train', color='blue')
    plt.plot(range(num_epoch), IoU_hist_val, label='IoU valid', color='red')
    plt.title('IoU durante el entrenamiento y validación')
    plt.xlabel('Épocas')
    plt.ylabel('IoU')
    plt.legend()

    # Graficar los datos de ocupación de las bboxes
    x = np.arange(len(box_vol_means))
    bar_width = 0.6

    print(f"volumen de los pólipos en train {box_vol_means[0]}")
    print(f"volumen de los pólipos en validación {box_vol_means[1]}")
    print(f"volumen de los pólipos en test {box_vol_means[2]}")

    plt.subplot(3, 2, 3)
    plt.bar(x, box_vol_means, width=bar_width, 
            color=colors[:len(eval_data)])
    plt.legend(["train", "validation", "test"])
    plt.title("Área promedio de pólipos por split")
    plt.ylabel("% de ocupación")

    plt.tight_layout()
    
    # Distribución de los centros de las bboxes
    names = ["train", "validation", "test"]
    for i, box_centers in enumerate(bbox_centers):
        plt.subplot(3, 2, 4+i)
        set_heat_point_map(plt.gca(), plt.gcf(), f"Centros de bboxes {names[i]}", box_centers, (1, 1))
    
    # guardamos la imagen si es necesario
    if save_img:
        plt.savefig(img_name, format='png', dpi=300)

    # Mostrar ambas gráficas
    plt.show()

    show_test_results(loss_test, IoU_test)




def graph_summary(idp):
    # Configuración del estilo de los gráficos
    sns.set(style="whitegrid")

    # Crear ventana con gráficos
    fig, axs = plt.subplots(4, 2, figsize=(8, 10))

    # Graficamos los diagramas
    charts = [
        # Gráfico 1: Distribución de las imágenes por split
        (idp.split_counts, axs[0, 0], 'División de imágenes del dataset', 'Número de Imágenes'),
        # Gráfico 2: Composición del dataset por tipo de luz
        (idp.light_counts, axs[0, 1], 'Composición del dataset por tipo de luz', 'Número de Imágenes'),
        # Gráfico 3: Tipos de resoluciones en las imágenes del dataset
        (idp.resolution_counts, axs[1, 0], 'Tipos de resoluciones en las imágenes del dataset', 'Número de Imágenes'),
        # Gráfico 4: Número de canales por tipo de imágen
        (idp.channel_counts, axs[1, 1], 'Formato de las imágenes', 'Número de Imágenes')
    ]

    for data, ax, title, ylabel in charts:
        ax.set_ylabel(ylabel)
        ax.bar(data.keys(), data.values(), color=['blue', 'green', 'orange'])
        ax.set_title(title)

    # Graficamos los histogramas
    hist = [
        # Gráfico 5: Histograma del brillo en las imágenes
        (idp.brightness, axs[2, 0], 'Brillo de los frames', 'Número de Imágenes'),
        # Gráfico 6: Histograma del contraste en las imágenes
        (idp.contrast, axs[2, 1], 'Contraste de los frames', 'Número de Imágenes')
    ]
    
    for data, ax, title, ylabel in hist:
        ax.hist(data, bins=20, color='forestgreen')
        ax.set_title(title)
        ax.set_ylabel(ylabel)


    # Gráfico 7: heatmap de distribución de las máscaras
    sns.heatmap(idp.mask_heatmap, cmap="crest", ax=axs[3, 0], cbar=True,
                xticklabels=False, yticklabels=False)
    axs[3, 0].set_title('Distribución de las máscaras (Heatmap)')
    axs[3, 0].set_box_aspect(idp.target_resolution[1] / idp.target_resolution[0])

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
    sc = axs[3, 1].scatter(cx, cy, c=z, s=5, cmap='YlOrRd', alpha=0.8)
    axs[3, 1].set_title('Densidad de centros de pólipos')
    axs[3, 1].set_xlim([0, idp.target_resolution[0]])
    axs[3, 1].set_ylim([0, idp.target_resolution[1]])
    axs[3, 1].set_box_aspect(idp.target_resolution[1] / idp.target_resolution[0])

    # Colorbar
    cbar = fig.colorbar(sc, ax=axs[3, 1])
    cbar.set_label('Densidad estimada')

    # Ajustar el layout
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
        charts[4][0].append(idp.sum_masks_percentage)
        charts[5][0].append(idp.sum_bbox_percentage)
        hist[0][0].append(idp.brightness)
        hist[1][0].append(idp.contrast)
    
    print(charts[4][0])
    print(charts[5][0])

    # 📊 graficos de barras 
    for data, ax, title, ylabel in charts:
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


def set_bar_graph(data_list, ax, title, ylabel, set_names):
    """
    Define un grafico de barras dentro de un mosaico de gráficos
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
            values = [d.get(k, 0) for k in all_keys]    # obtiene el dato para esa key, si no hay es 0

            # Desplazamos las barras para que no se superpongan
            ax.bar(x_pos + i * bar_width, values, bar_width, 
                    label=set_names[i])

        ax.set_xticks(x_pos + bar_width * (len(data_list) / 2 - 0.5))  # Centra las etiquetas
        ax.set_xticklabels(all_keys)  # Etiquetas para las barras
        ax.legend()
    else:   # para un dato un valor por barra
        x_pos = np.arange(len(data_list))
        ax.bar(x_pos, data_list, color=colors[:len(data_list)], 
               tick_label=set_names)



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

    
def show_image(idp, id, pred_bbox=None):
    """
    Esta funcion muestra la imagen con la bbox objetivo asociada y la predicha si dada
    Mostramos la imagen id si tenemos una prediccion de YOLO la muestra también
    """
    img_path = idp.dict[id]['path']
    img_w, img_h = idp.dict[id]['size']
    object_bbox = utils.bbox_cent2corn(idp.dict[id]['bbox'], img_w, img_h)
    img = mpimg.imread(img_path)

    print(f"Bbox objetivo: {object_bbox}")
    if pred_bbox is not None:
        pred_bbox = utils.bbox_cent2corn(idp.dict[id]['bbox'], img_w, img_h)
        print(f"Bbox predicha: {pred_bbox}")
    
    # Creamos la figura para añadir los datos
    fig, ax = plt.subplots(1, figsize=(10, 10))

    # Cargamos la imagen en la figura
    ax.imshow(img)

    # Añadimos el dibujo de la bbox objetivo
    x, y, w, h = object_bbox
    rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                            edgecolor='cyan', facecolor="none") 
    ax.add_patch(rect)
    
    # Añadimos el dibujo de la bbox predicha si dada
    if pred_bbox is not None:
        x, y, w, h = pred_bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                edgecolor='red', facecolor="none") 
        ax.add_patch(rect)
            
    # Add the patch to the Axes 
    plt.show()