import cv2
import torch
import numpy as np
from models import Darknet
from utils.utils import load_classes, non_max_suppression, rescale_boxes
from torchvision import transforms

# Cargar configuración y pesos
model_def = "config/yolov3.cfg"
weights_path = "weights/yolov3.weights"
class_path = "data/coco.names"  # Ruta al archivo que subiste

# Cargar el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(model_def, img_size=416).to(device)
model.load_darknet_weights(weights_path)
model.eval()

# Cargar clases
classes = load_classes(class_path)

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Transformación de imagen a tensor para YOLO
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((416, 416))
])

# Bucle principal de captura de video
frame_skip = 5  # Procesar solo 1 de cada 5 frames
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo obtener el frame de la cámara")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Detección de objetos
    RGBimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgTensor = transform(RGBimg).unsqueeze(0).to(device)

    with torch.no_grad():
        detections = model(imgTensor)
        detections = non_max_suppression(detections, conf_thres=0.5, nms_thres=0.4)

    # Dibujar los objetos detectados
    if detections[0] is not None:
        detections = rescale_boxes(detections[0], 416, frame.shape[:2])
        for *box, conf, cls_conf, cls_pred in detections:
            if cls_conf > 0.5:
                cls_pred = int(cls_pred)
                if cls_pred < len(classes):
                    label = f"{classes[cls_pred]} {cls_conf:.2f}"
                    x1, y1, x2, y2 = map(int, box)

                    # Dibujar el rectángulo y el texto
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    print(f"Índice fuera de rango: {cls_pred}")

    # Mostrar el video
    cv2.imshow("Video Feed", frame)

    # Redimensionar la ventana de visualización
    cv2.resizeWindow("Video Feed", 320, 240)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpiar
cap.release()
cv2.destroyAllWindows()
