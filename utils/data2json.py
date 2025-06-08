"""
    Este es un código para poder recuperar los datos de entrenamiento en caso de
    error usando la información del prompt
"""


import json

raw_data = [
    "Época 0  Loss train 0.0474  IoU train 0.5503", 
    "Época 0  Loss valid 0.0495  IoU valid 0.5306", 
    "Época 1  Loss train 0.0355  IoU train 0.6597", 
    "Época 1  Loss valid 0.0435  IoU valid 0.5859", 
    "Época 2  Loss train 0.0302  IoU train 0.7092", 
    "Época 2  Loss valid 0.0441  IoU valid 0.5770", 
    "Época 3  Loss train 0.0274  IoU train 0.7362", 
    "Época 3  Loss valid 0.0403  IoU valid 0.6165", 
    "Época 4  Loss train 0.0256  IoU train 0.7527", 
    "Época 4  Loss valid 0.0393  IoU valid 0.6237", 
    "Época 5  Loss train 0.0242  IoU train 0.7666", 
    "Época 5  Loss valid 0.0464  IoU valid 0.5590", 
    "Época 6  Loss train 0.0238  IoU train 0.7706", 
    "Época 6  Loss valid 0.0411  IoU valid 0.6078", 
    "Época 7  Loss train 0.0226  IoU train 0.7818", 
    "Época 7  Loss valid 0.0374  IoU valid 0.6427", 
    "Época 8  Loss train 0.0216  IoU train 0.7909", 
    "Época 8  Loss valid 0.0369  IoU valid 0.6508", 
    "Época 9  Loss train 0.0211  IoU train 0.7961", 
    "Época 9  Loss valid 0.0353  IoU valid 0.6636", 
    "Época 10  Loss train 0.0208  IoU train 0.7985", 
    "Época 10  Loss valid 0.0367  IoU valid 0.6495", 
    "Época 11  Loss train 0.0208  IoU train 0.7991", 
    "Época 11  Loss valid 0.0324  IoU valid 0.6909", 
    "Época 12  Loss train 0.0197  IoU train 0.8093", 
    "Época 12  Loss valid 0.0357  IoU valid 0.6590", 
    "Época 13  Loss train 0.0198  IoU train 0.8088", 
    "Época 13  Loss valid 0.0320  IoU valid 0.6935", 
    "Época 14  Loss train 0.0194  IoU train 0.8117", 
    "Época 14  Loss valid 0.0328  IoU valid 0.6870", 
    "Época 15  Loss train 0.0189  IoU train 0.8169", 
    "Época 15  Loss valid 0.0351  IoU valid 0.6652", 
    "Época 16  Loss train 0.0193  IoU train 0.8133", 
    "Época 16  Loss valid 0.0320  IoU valid 0.6923", 
    "Época 17  Loss train 0.0184  IoU train 0.8218", 
    "Época 17  Loss valid 0.0335  IoU valid 0.6803", 
    "Época 18  Loss train 0.0197  IoU train 0.8093", 
    "Época 18  Loss valid 0.0317  IoU valid 0.6991", 
    "Época 19  Loss train 0.0191  IoU train 0.8150", 
    "Época 19  Loss valid 0.0318  IoU valid 0.6947", 
    "Época 20  Loss train 0.0181  IoU train 0.8244", 
    "Época 20  Loss valid 0.0320  IoU valid 0.6951", 
    "Época 21  Loss train 0.0180  IoU train 0.8260", 
    "Época 21  Loss valid 0.0308  IoU valid 0.7078", 
    "Época 22  Loss train 0.0175  IoU train 0.8304", 
    "Época 22  Loss valid 0.0355  IoU valid 0.6629", 
    "Época 23  Loss train 0.0173  IoU train 0.8321", 
    "Época 23  Loss valid 0.0300  IoU valid 0.7120", 
    "Época 24  Loss train 0.0190  IoU train 0.8169", 
    "Época 24  Loss valid 0.0329  IoU valid 0.6880", 
    "Época 25  Loss train 0.0167  IoU train 0.8380", 
    "Época 25  Loss valid 0.0291  IoU valid 0.7227", 
    "Época 26  Loss train 0.0173  IoU train 0.8321", 
    "Época 26  Loss valid 0.0301  IoU valid 0.7151", 
    "Época 27  Loss train 0.0166  IoU train 0.8390", 
    "Época 27  Loss valid 0.0291  IoU valid 0.7222", 
    "Época 28  Loss train 0.0163  IoU train 0.8417", 
    "Época 28  Loss valid 0.0287  IoU valid 0.7255", 
    "Época 29  Loss train 0.0163  IoU train 0.8417", 
    "Época 29  Loss valid 0.0276  IoU valid 0.7373",]

loss_test = 0.019957
IoU_test = 0.811

loss_hist_train = []
IoU_hist_train = []
loss_hist_val = []
IoU_hist_val = []


for line in raw_data:
    split = line.split()
    if split[3] == "train":
        loss_hist_train.append(float(split[4]))
        IoU_hist_train.append(float(split[7]))
    if split[3] == "valid":
        loss_hist_val.append(float(split[4]))
        IoU_hist_val.append(float(split[7]))

json_dict = { 
    "loss_test": loss_test, 
    "IoU_test": IoU_test,
    "loss_hist_train": loss_hist_train,
    "IoU_hist_train": IoU_hist_train,
    "loss_hist_val": loss_hist_val,
    "IoU_hist_val": IoU_hist_val,
    "eval_data": None}

# guardamos los datos
with open("../resultados/Piccolo+CVC+PolypDB_DeiT_30EP_384x384.json", "w", encoding="utf-8") as json_file:
    json.dump(json_dict, json_file)  # `indent=4` para formato legible