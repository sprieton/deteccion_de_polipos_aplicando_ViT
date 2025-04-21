"""
    Este es un código para poder recuperar los datos de entrenamiento en caso de
    error usando la información del prompt
"""


import json

raw_data = [
    "Epoch 0  Loss train 0.0038  IoU train 0.2807",
    "Epoch 0  Loss valid 0.0019  IoU valid 0.3388",
    "Epoch 1  Loss train 0.0012  IoU train 0.3713",
    "Epoch 1  Loss valid 0.0015  IoU valid 0.4513",
    "Epoch 2  Loss train 0.0009  IoU train 0.4089",
    "Epoch 2  Loss valid 0.0016  IoU valid 0.4409",
    "Epoch 3  Loss train 0.0007  IoU train 0.4476",
    "Epoch 3  Loss valid 0.0014  IoU valid 0.4743",
    "Epoch 4  Loss train 0.0007  IoU train 0.4669",
    "Epoch 4  Loss valid 0.0016  IoU valid 0.5043",
    "Epoch 5  Loss train 0.0005  IoU train 0.5023",
    "Epoch 5  Loss valid 0.0014  IoU valid 0.4818",
    "Epoch 6  Loss train 0.0005  IoU train 0.5151",
    "Epoch 6  Loss valid 0.0014  IoU valid 0.5168",
    "Epoch 7  Loss train 0.0004  IoU train 0.5368",
    "Epoch 7  Loss valid 0.0013  IoU valid 0.5098",
    "Epoch 8  Loss train 0.0003  IoU train 0.5784",
    "Epoch 8  Loss valid 0.0014  IoU valid 0.4960",
    "Epoch 9  Loss train 0.0003  IoU train 0.5890",
    "Epoch 9  Loss valid 0.0014  IoU valid 0.5220",
    "Epoch 10  Loss train 0.0002  IoU train 0.6066", 
    "Epoch 10  Loss valid 0.0013  IoU valid 0.5264", 
    "Epoch 11  Loss train 0.0002  IoU train 0.6362", 
    "Epoch 11  Loss valid 0.0013  IoU valid 0.5116", 
    "Epoch 12  Loss train 0.0002  IoU train 0.6286", 
    "Epoch 12  Loss valid 0.0015  IoU valid 0.4610", 
    "Epoch 13  Loss train 0.0002  IoU train 0.6168", 
    "Epoch 13  Loss valid 0.0014  IoU valid 0.4838", 
    "Epoch 14  Loss train 0.0002  IoU train 0.6278", 
    "Epoch 14  Loss valid 0.0014  IoU valid 0.5180", 
    "Epoch 15  Loss train 0.0001  IoU train 0.6558", 
    "Epoch 15  Loss valid 0.0013  IoU valid 0.5226", 
    "Epoch 16  Loss train 0.0001  IoU train 0.6908", 
    "Epoch 16  Loss valid 0.0014  IoU valid 0.5260", 
    "Epoch 17  Loss train 0.0001  IoU train 0.7088", 
    "Epoch 17  Loss valid 0.0013  IoU valid 0.5292", 
    "Epoch 18  Loss train 0.0001  IoU train 0.7386", 
    "Epoch 18  Loss valid 0.0013  IoU valid 0.5360", 
    "Epoch 19  Loss train 0.0001  IoU train 0.7499", 
    "Epoch 19  Loss valid 0.0013  IoU valid 0.5309", 
    "Epoch 20  Loss train 0.0001  IoU train 0.7379", 
    "Epoch 20  Loss valid 0.0012  IoU valid 0.5361", 
    "Epoch 21  Loss train 0.0001  IoU train 0.7216", 
    "Epoch 21  Loss valid 0.0014  IoU valid 0.4804", 
    "Epoch 22  Loss train 0.0001  IoU train 0.7362", 
    "Epoch 22  Loss valid 0.0013  IoU valid 0.5218", 
    "Epoch 23  Loss train 0.0001  IoU train 0.7127", 
    "Epoch 23  Loss valid 0.0014  IoU valid 0.4881", 
    "Epoch 24  Loss train 0.0001  IoU train 0.7283", 
    "Epoch 24  Loss valid 0.0014  IoU valid 0.5239", 
    "Epoch 25  Loss train 0.0001  IoU train 0.7096", 
    "Epoch 25  Loss valid 0.0012  IoU valid 0.5504", 
    "Epoch 26  Loss train 0.0001  IoU train 0.7467", 
    "Epoch 26  Loss valid 0.0012  IoU valid 0.5268", 
    "Epoch 27  Loss train 0.0006  IoU train 0.5458", 
    "Epoch 27  Loss valid 0.0014  IoU valid 0.5538", 
    "Epoch 28  Loss train 0.0002  IoU train 0.6353", 
    "Epoch 28  Loss valid 0.0015  IoU valid 0.4811", 
    "Epoch 29  Loss train 0.0001  IoU train 0.6649", 
    "Epoch 29  Loss valid 0.0015  IoU valid 0.5243", 
    "Epoch 30  Loss train 0.0002  IoU train 0.6570", 
    "Epoch 30  Loss valid 0.0013  IoU valid 0.5244", 
    "Epoch 31  Loss train 0.0001  IoU train 0.7397", 
    "Epoch 31  Loss valid 0.0014  IoU valid 0.5349", 
    "Epoch 32  Loss train 0.0000  IoU train 0.7908", 
    "Epoch 32  Loss valid 0.0013  IoU valid 0.5293", 
    "Epoch 33  Loss train 0.0000  IoU train 0.8131", 
    "Epoch 33  Loss valid 0.0013  IoU valid 0.5302", 
    "Epoch 34  Loss train 0.0000  IoU train 0.8322", 
    "Epoch 34  Loss valid 0.0013  IoU valid 0.5282", 
    "Epoch 35  Loss train 0.0000  IoU train 0.8448", 
    "Epoch 35  Loss valid 0.0013  IoU valid 0.5378", 
    "Epoch 36  Loss train 0.0000  IoU train 0.8577", 
    "Epoch 36  Loss valid 0.0013  IoU valid 0.5417", 
    "Epoch 37  Loss train 0.0000  IoU train 0.8641", 
    "Epoch 37  Loss valid 0.0013  IoU valid 0.5448", 
    "Epoch 38  Loss train 0.0000  IoU train 0.8628", 
    "Epoch 38  Loss valid 0.0013  IoU valid 0.5311", 
    "Epoch 39  Loss train 0.0000  IoU train 0.8556", 
    "Epoch 39  Loss valid 0.0013  IoU valid 0.5311", 
    "Epoch 40  Loss train 0.0000  IoU train 0.8507", 
    "Epoch 40  Loss valid 0.0013  IoU valid 0.5397", 
    "Epoch 41  Loss train 0.0000  IoU train 0.8363", 
    "Epoch 41  Loss valid 0.0013  IoU valid 0.5403", 
    "Epoch 42  Loss train 0.0000  IoU train 0.7813", 
    "Epoch 42  Loss valid 0.0015  IoU valid 0.4934", 
    "Epoch 43  Loss train 0.0004  IoU train 0.5697", 
    "Epoch 43  Loss valid 0.0020  IoU valid 0.4801", 
    "Epoch 44  Loss train 0.0003  IoU train 0.6108", 
    "Epoch 44  Loss valid 0.0015  IoU valid 0.4830", 
    "Epoch 45  Loss train 0.0002  IoU train 0.6703", 
    "Epoch 45  Loss valid 0.0015  IoU valid 0.4909", 
    "Epoch 46  Loss train 0.0001  IoU train 0.6936", 
    "Epoch 46  Loss valid 0.0013  IoU valid 0.5539", 
    "Epoch 47  Loss train 0.0001  IoU train 0.7613", 
    "Epoch 47  Loss valid 0.0013  IoU valid 0.5274", 
    "Epoch 48  Loss train 0.0000  IoU train 0.7963", 
    "Epoch 48  Loss valid 0.0013  IoU valid 0.5453", 
    "Epoch 49  Loss train 0.0000  IoU train 0.8158", 
    "Epoch 49  Loss valid 0.0013  IoU valid 0.5500"]

loss_test = 0.0010723249211099812
IoU_test = 0.4330458655567454

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
    "IoU_hist_val": IoU_hist_val }

# guardamos los datos
with open("../resultados/Piccolo_YOLO-base_50EP_480x240.json", "w", encoding="utf-8") as json_file:
    json.dump(json_dict, json_file)  # `indent=4` para formato legible