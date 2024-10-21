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

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Transformación de imagen a tensor para YOLO
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((416, 416))
])

# Lista para almacenar puntos del polígono
points = []

# Función para manejar clics del mouse
def click_event(event, x, y, flags, param):
    global points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))  # Agrega el punto a la lista
        # Dibuja el punto en la imagen
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Punto rojo
        if len(points) > 2:  # Si hay al menos 3 puntos
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

# Asigna la función de clic al evento del mouse
cv2.namedWindow("Video Feed")
cv2.setMouseCallback("Video Feed", click_event)

# Bucle principal de captura de video
frame_skip = 5  # Procesar solo 1 de cada 5 frames
frame_count = 0
unique_id_counter = 0  # Contador para ID únicos
detected_ids = {}  # Diccionario para almacenar IDs de objetos detectados
previous_detections = {}  # Diccionario para el seguimiento de objetos

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

    # Inicializar un diccionario para contar objetos detectados
    detected_counts = {}

    # Obtener los puntos del polígono para la detección
    if len(points) > 2:
        polygon = np.array(points, dtype=np.int32)

    # Dibujar los objetos detectados y verificar si están dentro del polígono
    if detections[0] is not None:
        detections = rescale_boxes(detections[0], 416, frame.shape[:2])
        for *box, conf, cls_conf, cls_pred in detections:
            if cls_conf > 0.5:
                cls_pred = int(cls_pred)
                if cls_pred < len(classes):
                    label = f"{classes[cls_pred]} {cls_conf:.2f}"
                    x1, y1, x2, y2 = map(int, box)

                    # Verifica si el centro del objeto está dentro del polígono
                    center = np.array([(x1 + x2) // 2, (y1 + y2) // 2], dtype=np.float32)
                    if len(points) > 2 and cv2.pointPolygonTest(polygon, tuple(center), False) >= 0:
                        # Mantener el ID si ya existe
                        unique_id = None
                        for uid, (prev_box, _) in previous_detections.items():
                            prev_center = np.array([(prev_box[0] + prev_box[2]) // 2, (prev_box[1] + prev_box[3]) // 2])
                            if np.linalg.norm(prev_center - center) < 50:  # Distancia umbral
                                unique_id = uid
                                break
                        if unique_id is None:
                            unique_id = unique_id_counter
                            unique_id_counter += 1

                        # Almacenar la detección
                        previous_detections[unique_id] = (box, classes[cls_pred])

                        # Dibuja el rectángulo y etiqueta
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} ID: {unique_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Contar el objeto detectado
                        detected_counts[classes[cls_pred]] = detected_counts.get(classes[cls_pred], 0) + 1

    # Limpiar detecciones fuera del polígono
    for uid in list(previous_detections.keys()):
        box, _ = previous_detections[uid]
        center = np.array([(box[0] + box[2]) // 2, (box[1] + box[3]) // 2])
        if len(points) > 2 and cv2.pointPolygonTest(polygon, tuple(center), False) < 0:
            del previous_detections[uid]  # Eliminar si salió del polígono

    # Mostrar conteo de objetos detectados
    y_offset = 30
    for class_name, count in detected_counts.items():
        cv2.putText(frame, f"{class_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += 20

    # Dibujar el polígono en la imagen
    if len(points) > 2:
        cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)

    # Mostrar el video
    cv2.imshow("Video Feed", frame)

    # Redimensionar la ventana de visualización
    cv2.resizeWindow("Video Feed", 320, 240)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpiar
cap.release()
cv2.destroyAllWindows()
