import cv2
import numpy as np
import torch
from models import Darknet
from utils.utils import load_classes, non_max_suppression, rescale_boxes
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F  # Para el redimensionamiento con interpolate

# Función para hacer padding y convertir a una imagen cuadrada
def pad_to_square(img, pad_value):
    _, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (bottom / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding for height and width
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = torch.nn.functional.pad(img, pad, "constant", value=pad_value)
    return img, pad

# Función para redimensionar imágenes utilizando PyTorch
def resize(img, size):
    img = F.interpolate(img.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return img

# Variables para almacenar los puntos del polígono
polygon_points = []
is_drawing = False
mode = 'create'  # Modo inicial: 'create' para crear puntos

# Función de callback para dibujar el polígono
def draw_polygon(event, x, y, flags, param):
    global polygon_points, is_drawing, mode

    if mode == 'create':
        # Crear puntos
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon_points.append((x, y))
            is_drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            is_drawing = False

# Reemplazar por tu URL RTSP
rtsp_url = "rtsp://admin:@Plaza1234@192.168.4.10:554/channel=1_stream=0.sdp?real_stream"

# Inicializar YOLO
model_def = "config/yolov3-tiny.cfg"
weights_path = "weights/yolov3-tiny.weights"
class_path = "data/coco.names"
conf_thres = 0.8
nms_thres = 0.4
img_size = 416

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(model_def, img_size=img_size).to(device)

# Cargar pesos
model.load_darknet_weights(weights_path)
model.eval()

# Cargar clases
classes = load_classes(class_path)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Inicializar la cámara IP
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("No se pudo conectar a la cámara IP")
    exit()

cv2.namedWindow("Video Feed")
cv2.setMouseCallback("Video Feed", draw_polygon)

# Definir la nueva resolución (ancho, alto)
new_width, new_height = 320, 240  # Resolución más baja para mejor rendimiento
cap.set(cv2.CAP_PROP_FPS, 30)

frame_count = 0  # Contador de frames para la detección

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    # Incrementar el contador de frames
    frame_count += 1

    # Redimensionar el frame
    frame = cv2.resize(frame, (new_width, new_height))

    # Detección cada 5 frames para optimizar el rendimiento
    if frame_count % 5 == 0:
        # Convertir la imagen a RGB para YOLO
        RGBimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, img_size)  # Redimensionar la imagen
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))

        # Detección de objetos
        with torch.no_grad():
            detections = model(imgTensor)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Procesar detecciones
        if detections[0] is not None:
            detections = detections[0]  # Solo procesar la primera imagen de las detecciones
            detections = rescale_boxes(detections, img_size, frame.shape[:2])  # Reescalar las cajas a la imagen original

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                label = f"{classes[int(cls_pred)]} {cls_conf:.2f}"
                # Dibuja el rectángulo alrededor del objeto
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el video en tiempo real
    cv2.imshow("Video Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Salir si se presiona 'q'
        break

cap.release()
cv2.destroyAllWindows()