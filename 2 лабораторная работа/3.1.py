import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Загрузка обученной модели
model = YOLO('best.pt')  # Путь к файлу с весами

# Инициализация видеопотока
cap = cv2.VideoCapture(0)  # 0 - встроенная камера, или указать путь к видеофайлу

# Инициализация аннотаторов
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Выполнение детекции
    results = model(frame)[0]
    
    # Преобразование результатов в формат supervision
    detections = sv.Detections.from_ultralytics(results)
    
    # Аннотирование кадра
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame, 
        detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, 
        detections=detections
    )
    
    # Отображение результатов
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
