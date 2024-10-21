import cv2
import torch
import numpy as np
from models import Darknet
from utils.utils import load_classes, non_max_suppression, rescale_boxes
from torchvision import transforms
from sort import Sort  # Algoritmo de seguimiento SORT
import tkinter as tk
from tkinter import simpledialog, messagebox

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

# Inicializar el rastreador SORT
tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

# Variables para almacenar los puntos del polígono y las líneas
points = []
lines = []

# Función para solicitar ID de línea mediante una ventana emergente
def get_line_id():
    root = tk.Tk()
    line_id = simpledialog.askstring("Input", "Ingrese el ID de la línea:", parent=root)
    return line_id

# Función para manejar las entradas
def handle_entry():
    if lines:
        for line in lines:
            line["color"] = (0, 255, 0)  # Cambiar color a verde para entradas
        messagebox.showinfo("Entrada", "IDs de entrada asignados a las líneas.")
    else:
        messagebox.showwarning("Entrada", "No hay líneas para asignar IDs.")

# Función para manejar las salidas
def handle_exit():
    if lines:
        for line in lines:
            line["color"] = (0, 0, 255)  # Cambiar color a rojo para salidas
        messagebox.showinfo("Salida", "IDs de salida asignados a las líneas.")
    else:
        messagebox.showwarning("Salida", "No hay líneas para asignar IDs.")

# Evento de mouse para dibujar polígono y líneas
def draw_polygon(event, x, y, flags, param):
    global points, lines
    if event == cv2.EVENT_LBUTTONDOWN:
        # Agregar un punto al polígono
        points.append((x, y))
        
        if len(points) >= 2:
            # Solicitar ID de la línea solo después de tener al menos 2 puntos
            line_id = get_line_id()
            if line_id:  # Solo proceder si se ingresó un ID
                # Crear una línea entre los últimos dos puntos
                line = {
                    "start": points[-2],  # Punto A
                    "end": points[-1],    # Punto B
                    "id": line_id,        # ID de la línea
                    "color": (255, 0, 0)  # Color inicial en azul
                }
                lines.append(line)  # Agregar la línea a la lista

# Función para verificar si un punto está dentro del polígono
def is_point_in_polygon(point, polygon):
    if len(polygon) < 3:  # Verificar que haya al menos 3 puntos en el polígono
        return False
    polygon_np = np.array(polygon, dtype=np.int32)  # Convertir a un array de NumPy
    return cv2.pointPolygonTest(polygon_np, point, False) >= 0

# Configurar la ventana de OpenCV
cv2.namedWindow("Video Feed")
cv2.setMouseCallback("Video Feed", draw_polygon)

# Crear la interfaz de usuario con botones
root = tk.Tk()
root.title("Control de Video")
frame = tk.Frame(root)
frame.pack(side=tk.BOTTOM)

entry_button = tk.Button(frame, text="Entrada", command=handle_entry)
entry_button.pack(side=tk.LEFT)

exit_button = tk.Button(frame, text="Salida", command=handle_exit)
exit_button.pack(side=tk.LEFT)

# Bucle principal de captura de video
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

    # Almacenar las detecciones formateadas para SORT
    detections_for_sort = []
    class_labels_for_sort = []

    if detections[0] is not None:
        detections = rescale_boxes(detections[0], 416, frame.shape[:2])
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            center_x, center_y = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            detections_for_sort.append([x1, y1, x2, y2, conf.item()])
            class_labels_for_sort.append(int(cls_pred))

    # Manejo de detecciones vacías
    if len(detections_for_sort) == 0:
        tracked_objects = tracker.update(np.empty((0, 5)))  # Actualizar sin detecciones
    else:
        # Solo mantener detecciones dentro del polígono
        filtered_detections = []
        for detection in detections_for_sort:
            x1, y1, x2, y2, conf = detection
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if len(points) >= 3 and is_point_in_polygon(center, points):  # Verificar si hay un polígono válido
                filtered_detections.append(detection)

        tracked_objects = tracker.update(np.array(filtered_detections))

    # Dibujar los objetos rastreados
    for i, track in enumerate(tracked_objects):
        x1, y1, x2, y2, track_id = track
        class_label = classes[class_labels_for_sort[i]] if i < len(class_labels_for_sort) else "Unknown"
        label = f"ID: {int(track_id)} {class_label}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Dibujar el polígono si tiene suficientes puntos
    if len(points) >= 2:
        for line in lines:
            cv2.line(frame, line["start"], line["end"], line["color"], 2)  # Dibujar la línea con el color asignado
            # Mostrar el ID de la línea al costado
            mid_point = ((line["start"][0] + line["end"][0]) // 2, (line["start"][1] + line["end"][1]) // 2)
            cv2.putText(frame, f"ID: {line['id']}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, line["color"], 1)

    # Mostrar el video
    cv2.imshow("Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpiar
cap.release()
cv2.destroyAllWindows()