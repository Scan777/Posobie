import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Загрузка модели
model = YOLO('best.pt')

# Инициализация трекера
tracker = sv.ByteTrack()

# Инициализация аннотаторов
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Детекция объектов
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Обновление трекера
    detections = tracker.update_with_detections(detections)
    
    # Аннотирование кадра с трекингом
    annotated_frame = trace_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )
    
    # Формирование меток с ID трекера
    labels = [
        f"#{tracker_id} {model.names[class_id]} {confidence:.2f}"
        for tracker_id, class_id, confidence
        in zip(detections.tracker_id, detections.class_id, detections.confidence)
    ]
    
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )
    
    # Вывод количества отслеживаемых объектов
    cv2.putText(annotated_frame, f"Tracking: {len(detections)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('YOLOv8 Object Tracking', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
