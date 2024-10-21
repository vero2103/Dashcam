#OPTIMIZACION DE RAPIDEZ DE VIDEO CON YOLOV3

import cv2
import torch
import numpy as np
from models import Darknet
from utils.utils import load_classes, non_max_suppression, rescale_boxes
from torchvision import transforms

# Cargar configuración y pesos
model_def = "config/yolov3.cfg"
weights_path = "weights/yolov3.weights"
class_path = "data/coco.names"

# Cargar el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(model_def, img_size=416).to(device)
model.load_darknet_weights(weights_path)
model.eval()

# Cargar clases
classes = load_classes(class_path)

# Definir la URL RTSP de la cámara IP
#rtsp_url = "rtsp://admin:@Plaza1234@192.168.4.10:554/channel=1_stream=0.sdp?real_stream"
#cap1 = cv2.VideoCapture()  # Webcam por defecto

# Inicializar la cámara IP
cap = cv2.VideoCapture(0)

# Transformación de imagen a tensor para YOLO
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((416, 416))
])

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo obtener el frame de la cámara IP")
        break

    # Procesar el frame
    RGBimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgTensor = transform(RGBimg).unsqueeze(0).to(device)

    # Detección
    with torch.no_grad():
        detections = model(imgTensor)
        detections = non_max_suppression(detections, conf_thres=0.8, nms_thres=0.4)

    # Procesar detecciones
    if detections[0] is not None:
        detections = rescale_boxes(detections[0], 416, frame.shape[:2])
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            label = f"{classes[int(cls_pred)]} {cls_conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el video
    cv2.imshow("Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()