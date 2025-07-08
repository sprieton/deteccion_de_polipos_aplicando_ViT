"""
    Este código proporciona una herramienta para poder guardar los datos de 
    salida de un modelo evaluado en caso de que ocurra algún error en entrenamiento
    o por conveniencia del usuario.
"""


import json

raw_data = [
    "Época 0  Loss train 0.0790  IoU train 0.0374 ",
    "Época 0  Loss valid 0.0110  IoU valid 0.1383 ",
    "Época 1  Loss train 0.0077  IoU train 0.1670 ",
    "Época 1  Loss valid 0.0019  IoU valid 0.2764 ",
    "Época 2  Loss train 0.0023  IoU train 0.2962 ",
    "Época 2  Loss valid 0.0017  IoU valid 0.3308 ",
    "Época 3  Loss train 0.0016  IoU train 0.3443 ",
    "Época 3  Loss valid 0.0017  IoU valid 0.3359 ",
    "Época 4  Loss train 0.0014  IoU train 0.3732 ",
    "Época 4  Loss valid 0.0022  IoU valid 0.3499 ",
    "Época 5  Loss train 0.0013  IoU train 0.3932 ",
    "Época 5  Loss valid 0.0015  IoU valid 0.3852 ",
    "Época 6  Loss train 0.0010  IoU train 0.4291 ",
    "Época 6  Loss valid 0.0016  IoU valid 0.3565 ",
    "Época 7  Loss train 0.0009  IoU train 0.4424 ",
    "Época 7  Loss valid 0.0014  IoU valid 0.3805 ",
    "Época 8  Loss train 0.0008  IoU train 0.4569 ",
    "Época 8  Loss valid 0.0015  IoU valid 0.3968 ",
    "Época 9  Loss train 0.0007  IoU train 0.4723 ",
    "Época 9  Loss valid 0.0013  IoU valid 0.3999 ",
    "Época 10  Loss train 0.0007  IoU train 0.4775 ",
    "Época 10  Loss valid 0.0015  IoU valid 0.3818 ",
    "Época 11  Loss train 0.0006  IoU train 0.5049 ",
    "Época 11  Loss valid 0.0013  IoU valid 0.4197 ",
    "Época 12  Loss train 0.0005  IoU train 0.5233 ",
    "Época 12  Loss valid 0.0014  IoU valid 0.3566 ",
    "Época 13  Loss train 0.0007  IoU train 0.4782 ",
    "Época 13  Loss valid 0.0014  IoU valid 0.4028 ",
    "Época 14  Loss train 0.0005  IoU train 0.5079 ",
    "Época 14  Loss valid 0.0013  IoU valid 0.3932 ",
    "Época 15  Loss train 0.0005  IoU train 0.5165 ",
    "Época 15  Loss valid 0.0015  IoU valid 0.3566 ",
    "Época 16  Loss train 0.0006  IoU train 0.4817 ",
    "Época 16  Loss valid 0.0013  IoU valid 0.4046 ",
    "Época 17  Loss train 0.0004  IoU train 0.5521 ",
    "Época 17  Loss valid 0.0014  IoU valid 0.3346 ",
    "Época 18  Loss train 0.0003  IoU train 0.5621 ",
    "Época 18  Loss valid 0.0012  IoU valid 0.4140 ",
    "Época 19  Loss train 0.0002  IoU train 0.6025 ",
    "Época 19  Loss valid 0.0012  IoU valid 0.4223 ",
    "Época 20  Loss train 0.0003  IoU train 0.5823 ",
    "Época 20  Loss valid 0.0015  IoU valid 0.3602 ",
    "Época 21  Loss train 0.0002  IoU train 0.5999 ",
    "Época 21  Loss valid 0.0016  IoU valid 0.3451 ",
    "Época 22  Loss train 0.0003  IoU train 0.5839 ",
    "Época 22  Loss valid 0.0012  IoU valid 0.4096 ",
    "Época 23  Loss train 0.0002  IoU train 0.6332 ",
    "Época 23  Loss valid 0.0012  IoU valid 0.3921 ",
    "Época 24  Loss train 0.0002  IoU train 0.6388 ",
    "Época 24  Loss valid 0.0012  IoU valid 0.4107 ",
    "Época 25  Loss train 0.0002  IoU train 0.6489 ",
    "Época 25  Loss valid 0.0011  IoU valid 0.4141 ",
    "Época 26  Loss train 0.0001  IoU train 0.6609 ",
    "Época 26  Loss valid 0.0011  IoU valid 0.4456 ",
    "Época 27  Loss train 0.0002  IoU train 0.6539 ",
    "Época 27  Loss valid 0.0012  IoU valid 0.4280 ",
    "Época 28  Loss train 0.0001  IoU train 0.6785 ",
    "Época 28  Loss valid 0.0012  IoU valid 0.4038 ",
    "Época 29  Loss train 0.0001  IoU train 0.6883 ",
    "Época 29  Loss valid 0.0012  IoU valid 0.3879",
]

loss_test = 0.001119
IoU_test = 0.333

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
with open("../resultados/Piccolo_YOLOv8_30EP_384x384.json", "w", encoding="utf-8") as json_file:
    json.dump(json_dict, json_file)  # `indent=4` para formato legible