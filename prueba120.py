import cv2
import torch
import numpy as np
from models import Darknet
from utils.utils import load_classes, non_max_suppression, rescale_boxes
from torchvision import transforms
import tkinter as tk
import threading

# Crear un diccionario para almacenar el color de las líneas
line_colors = {}

# Función para manejar la entrada
def handle_entry():
    line_id = entry_id.get()
    if line_id.isdigit() and int(line_id) in line_ids:
        line_colors[int(line_id)] = (0, 0, 0)  # Verde para entrada
        print("Entrada registrada para línea ID:", line_id)
    else:
        print("ID de línea no válido")

# Función para manejar la salida
def handle_exit():
    line_id = entry_id.get()
    if line_id.isdigit() and int(line_id) in line_ids:
        line_colors[int(line_id)] = (0, 0, 255)  # Rojo para salida
        print("Salida registrada para línea ID:", line_id)
    else:
        print("ID de línea no válido")

# Crear la interfaz de usuario
def create_ui():
    global entry_id
    root = tk.Tk()
    root.title("Control de Entradas y Salidas")
    
    entry_id = tk.Entry(root)
    entry_id.pack(pady=10)
    entry_id.insert(0, "Ingresa ID de línea")

    entry_button = tk.Button(root, text="Entrada", command=handle_entry)
    entry_button.pack(pady=10)

    exit_button = tk.Button(root, text="Salida", command=handle_exit)
    exit_button.pack(pady=10)

    root.mainloop()

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

# Lista para almacenar puntos del polígono y un diccionario para IDs de líneas
points = []
line_ids = {}
line_id_counter = 0

# Función para manejar clics del mouse
def click_event(event, x, y, flags, param):
    global points, frame, line_id_counter, line_ids
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        if len(points) > 1:
            line_ids[line_id_counter] = (points[-2], points[-1])
            line_colors[line_id_counter] = (255, 0, 0)  # Color inicial
            cv2.line(frame, points[-2], points[-1], (255, 0, 0), 2)
            line_id_counter += 1

        if len(points) > 2:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

# Asigna la función de clic al evento del mouse
cv2.namedWindow("Video Feed")
cv2.setMouseCallback("Video Feed", click_event)

# Crear la interfaz de usuario en un hilo separado
ui_thread = threading.Thread(target=create_ui)
ui_thread.daemon = True  # Permite que el hilo se cierre cuando el programa principal lo haga
ui_thread.start()

# Bucle principal de captura de video
frame_skip = 5
frame_count = 0
unique_id_counter = 0
detected_ids = {}
previous_detections = {}

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

    detected_counts = {}
    if len(points) > 2:
        polygon = np.array(points, dtype=np.int32)

    if detections[0] is not None:
        detections = rescale_boxes(detections[0], 416, frame.shape[:2])
        for *box, conf, cls_conf, cls_pred in detections:
            if cls_conf > 0.5:
                cls_pred = int(cls_pred)
                if cls_pred < len(classes):
                    label = f"{classes[cls_pred]} {cls_conf:.2f}"
                    x1, y1, x2, y2 = map(int, box)

                    center = np.array([(x1 + x2) // 2, (y1 + y2) // 2], dtype=np.float32)
                    if len(points) > 2 and cv2.pointPolygonTest(polygon, tuple(center), False) >= 0:
                        unique_id = None
                        for uid, (prev_box, _) in previous_detections.items():
                            prev_center = np.array([(prev_box[0] + prev_box[2]) // 2, (prev_box[1] + prev_box[3]) // 2])
                            if np.linalg.norm(prev_center - center) < 50:
                                unique_id = uid
                                break
                        if unique_id is None:
                            unique_id = unique_id_counter
                            unique_id_counter += 1

                        previous_detections[unique_id] = (box, classes[cls_pred])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} ID: {unique_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        detected_counts[classes[cls_pred]] = detected_counts.get(classes[cls_pred], 0) + 1

    for uid in list(previous_detections.keys()):
        box, _ = previous_detections[uid]
        center = np.array([(box[0] + box[2]) // 2, (box[1] + box[3]) // 2])
        if len(points) > 2 and cv2.pointPolygonTest(polygon, tuple(center), False) < 0:
            del previous_detections[uid]

    y_offset = 30
    for class_name, count in detected_counts.items():
        cv2.putText(frame, f"{class_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += 20

    if len(points) > 2:
        cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)

    # Dibujar líneas con sus colores correspondientes
    for line_id, (start, end) in line_ids.items():
        color = line_colors.get(line_id, (255, 0, 0))  # Rojo como color predeterminado
        cv2.line(frame, start, end, color, 2)
        cv2.putText(frame, f"Line ID: {line_id}", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Video Feed", frame)
    cv2.resizeWindow("Video Feed", 320, 240)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpiar
cap.release()
cv2.destroyAllWindows()
