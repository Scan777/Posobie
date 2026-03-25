import cv2
import numpy as np
import time
from ultralytics import YOLO
import supervision as sv
from Kuka import Kuka
from openshowvar import openshowvar

class YOLORobotControl:
    """Класс для управления роботом на основе YOLO детектирования"""
    
    def __init__(self, model_path='best.pt', robot_ip='192.168.1.2', port=7000):
        # Подключение к роботу
        self.robot = openshowvar(ip=robot_ip, port=port)
        self.kuka = Kuka(self.robot)
        
        # Загрузка модели YOLO
        self.model = YOLO(model_path)
        
        # Инициализация камеры
        self.cap = cv2.VideoCapture(0)
        
        # Калибровочные параметры
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Загрузка параметров калибровки (если есть)
        try:
            with np.load('calib.npz') as data:
                self.camera_matrix = data['mtx']
                self.dist_coeffs = data['dist']
            print("Параметры калибровки загружены")
        except:
            print("Калибровка не найдена, работа без коррекции искажений")
        
        # Параметры рабочей зоны
        self.workspace = {
            'x_min': 200, 'x_max': 600,
            'y_min': 100, 'y_max': 500,
            'z_pick': 50,      # высота захвата
            'z_approach': 100, # высота подхода
            'pixel_to_mm': 0.5 # масштабный коэффициент
        }
        
        # Инициализация трекера
        self.tracker = sv.ByteTrack()
        
        # Аннотаторы для визуализации
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
    
    def pixel_to_robot_coords(self, x_pixel, y_pixel):
        """Преобразование пиксельных координат в координаты робота"""
        robot_x = x_pixel * self.workspace['pixel_to_mm'] + self.workspace['x_min']
        robot_y = y_pixel * self.workspace['pixel_to_mm'] + self.workspace['y_min']
        return robot_x, robot_y
    
    def detect_objects(self):
        """Детектирование объектов с помощью YOLO"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None
        
        # Коррекция искажений (если есть калибровка)
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h)
            )
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        
        # Детекция объектов
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Обновление трекера
        detections = self.tracker.update_with_detections(detections)
        
        return detections, frame, results
    
    def move_to_object(self, detection):
        """Перемещение робота к обнаруженному объекту"""
        
        # Получение координат bounding box
        x1, y1, x2, y2 = detection.astype(int)
        
        # Вычисление центра объекта
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Преобразование в координаты робота
        robot_x, robot_y = self.pixel_to_robot_coords(cx, cy)
        
        print(f"Обнаружен объект в пиксельных координатах: ({cx}, {cy})")
        print(f"Координаты для робота: X={robot_x:.1f}, Y={robot_y:.1f}")
        
        # Установка параметров движения
        self.kuka.set_base(1)
        self.kuka.set_tool(1)
        self.kuka.set_speed(30)
        
        # Точка подхода
        approach_point = np.array([
            robot_x,
            robot_y,
            self.workspace['z_approach'],
            -180, 0, 0
        ])
        
        # Точка захвата
        pick_point = np.array([
            robot_x,
            robot_y,
            self.workspace['z_pick'],
            -180, 0, 0
        ])
        
        # Движение к объекту
        self.kuka.lin_continuous(self.kuka, np.array([approach_point]))
        self.kuka.lin_continuous(self.kuka, np.array([pick_point]))
        
        return robot_x, robot_y
    
    def run(self):
        """Основной цикл работы"""
        
        # Предустановленное место размещения
        place_position = (500, 400)
        
        try:
            while True:
                # Детектирование объектов
                detections, frame, results = self.detect_objects()
                
                if detections and len(detections) > 0:
                    # Формирование меток для отображения
                    labels = [
                        f"#{tracker_id} {self.model.names[class_id]} {confidence:.2f}"
                        for tracker_id, class_id, confidence
                        in zip(detections.tracker_id, detections.class_id, detections.confidence)
                    ]
                    
                    # Аннотирование кадра
                    annotated_frame = self.bounding_box_annotator.annotate(
                        scene=frame.copy(),
                        detections=detections
                    )
                    annotated_frame = self.label_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections,
                        labels=labels
                    )
                    
                    # Дополнительная информация
                    cv2.putText(annotated_frame, f"Objects: {len(detections)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "Press 'p' to pick first object", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    
                    # Координаты первого объекта
                    if len(detections) > 0:
                        x1, y1, x2, y2 = detections.xyxy[0].astype(int)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        robot_x, robot_y = self.pixel_to_robot_coords(cx, cy)
                        
                        cv2.putText(annotated_frame, 
                                   f"Robot: X={robot_x:.1f}, Y={robot_y:.1f}", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (255, 255, 0), 2)
                else:
                    annotated_frame = frame
                    cv2.putText(annotated_frame, "No objects detected", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 0, 255), 2)
                
                cv2.imshow('YOLO Robot Control', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p') and detections and len(detections) > 0:
                    # Захват и перемещение первого объекта
                    robot_x, robot_y = self.move_to_object(detections.xyxy[0])
                    
                    # Захват
                    self.kuka.close_grip()
                    time.sleep(1)
                    
                    # Подъем
                    approach_point = np.array([
                        robot_x,
                        robot_y,
                        self.workspace['z_approach'],
                        -180, 0, 0
                    ])
                    self.kuka.lin_continuous(self.kuka, np.array([approach_point]))
                    
                    # Перемещение к месту размещения
                    place_point = np.array([
                        place_position[0],
                        place_position[1],
                        self.workspace['z_approach'],
                        -180, 0, 0
                    ])
                    self.kuka.lin_continuous(self.kuka, np.array([place_point]))
                    
                    # Опускание
                    place_low = np.array([
                        place_position[0],
                        place_position[1],
                        self.workspace['z_pick'],
                        -180, 0, 0
                    ])
                    self.kuka.lin_continuous(self.kuka, np.array([place_low]))
                    
                    # Освобождение
                    self.kuka.open_grip()
                    time.sleep(1)
                    
                    # Подъем
                    self.kuka.lin_continuous(self.kuka, np.array([place_point]))
                    
                    print("Операция pick-and-place завершена")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.kuka.quit()

if __name__ == "__main__":
    controller = YOLORobotControl(model_path='best.pt', robot_ip='192.168.17.2')
    controller.run()
