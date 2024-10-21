import cv2
import torch
import numpy as np
from models import Darknet
from utils.utils import load_classes, non_max_suppression, rescale_boxes
from torchvision import transforms
from sort import Sort  # Asegúrate de tener la implementación de SORT

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

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Transformación de imagen a tensor para YOLO
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((416, 416))
])

# Inicializar el objeto SORT
tracker = Sort()

# Variables para almacenar los puntos del polígono
points = []

# Evento de mouse para dibujar polígono
def draw_polygon(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

# Configurar la ventana de OpenCV
cv2.namedWindow("Video Feed")
cv2.setMouseCallback("Video Feed", draw_polygon)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo obtener el frame de la cámara")
        break

    # Procesar el frame
    RGBimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgTensor = transform(RGBimg).unsqueeze(0).to(device)

    # Detección
    with torch.no_grad():
        detections = model(imgTensor)
        detections = non_max_suppression(detections, conf_thres=0.8, nms_thres=0.4)

    if detections[0] is not None:
        detections = rescale_boxes(detections[0], 416, frame.shape[:2])
        
        # Formato para SORT: [x1, y1, x2, y2, score]
        detections_for_sort = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            detections_for_sort.append([x1, y1, x2, y2, conf])
        
        detections_for_sort = np.array(detections_for_sort)

        # Actualizar el rastreador
        trackers = tracker.update(detections_for_sort)

        # Dibujar los resultados
        for i in range(len(trackers)):
            x1, y1, x2, y2, track_id = trackers[i]
            label = f"ID: {int(track_id)}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Dibujar el polígono dinámico si hay suficientes puntos
    if len(points) >= 3:
        cv2.polylines(frame, [np.array(points)], isClosed=False, color=(255, 0, 0), thickness=2)

    # Mostrar el video
    cv2.imshow("Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpiar
cap.release()
cv2.destroyAllWindows()
