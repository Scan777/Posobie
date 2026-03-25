import cv2
import numpy as np
import time
from Kuka import Kuka
from openshowvar import openshowvar

class ColorDetectionRobot:
    """Класс для управления роботом на основе цветового детектирования"""
    
    def __init__(self, robot_ip='192.168.1.2', port=7000):
        # Подключение к роботу
        self.robot = openshowvar(ip=robot_ip, port=port)
        self.kuka = Kuka(self.robot)
        
        # Параметры цветового детектирования (из ЛР №3)
        self.color_ranges = {
            'red': [(0, 158, 0), (16, 255, 255)],
            'yellow': [(17, 158, 0), (35, 255, 255)],
            'green': [(36, 158, 0), (52, 255, 255)],
            'blue': [(66, 158, 0), (108, 255, 255)]
        }
        
        # Параметры преобразования координат
        self.pixel_to_mm_x = 0.5  # мм/пиксель
        self.pixel_to_mm_y = 0.5
        self.offset_x = 300  # смещение базовой системы
        self.offset_y = 200
        
        # Инициализация камеры
        self.cap = cv2.VideoCapture(0)
        
        # Рабочая зона (определяется экспериментально)
        self.workspace = {
            'x_min': 200, 'x_max': 600,
            'y_min': 100, 'y_max': 500,
            'z_pick': 50,   # высота захвата
            'z_place': 20   # высота размещения
        }
    
    def detect_objects(self, target_color='red', min_area=500):
        """Детектирование объектов заданного цвета"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        # Преобразование в HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Создание маски
        lower = np.array(self.color_ranges[target_color][0])
        upper = np.array(self.color_ranges[target_color][1])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Морфологические операции
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Преобразование в координаты робота
                    robot_x = cx * self.pixel_to_mm_x + self.offset_x
                    robot_y = cy * self.pixel_to_mm_y + self.offset_y
                    
                    objects.append({
                        'pixel_center': (cx, cy),
                        'robot_coords': (robot_x, robot_y),
                        'area': area,
                        'contour': cnt
                    })
        
        return objects, frame
    
    def pick_and_place(self, object_coords, place_coords):
        """Выполнение операции захвата и перемещения"""
        
        # Установка базы и инструмента
        self.kuka.set_base(1)  # База рабочей зоны
        self.kuka.set_tool(1)  # Захват
        self.kuka.set_speed(50)  # Скорость 50%
        
        # Точка подхода (над объектом)
        approach_point = np.array([
            object_coords[0],
            object_coords[1],
            self.workspace['z_approach'],
            -180, 0, 0
        ])
        
        # Точка захвата
        pick_point = np.array([
            object_coords[0],
            object_coords[1],
            self.workspace['z_pick'],
            -180, 0, 0
        ])
        
        # Точка подхода к месту размещения
        approach_place = np.array([
            place_coords[0],
            place_coords[1],
            self.workspace['z_approach'],
            -180, 0, 0
        ])
        
        # Точка размещения
        place_point = np.array([
            place_coords[0],
            place_coords[1],
            self.workspace['z_place'],
            -180, 0, 0
        ])
        
        # Выполнение движения
        self.kuka.lin_continuous(self.kuka, np.array([approach_point]))
        self.kuka.lin_continuous(self.kuka, np.array([pick_point]))
        
        # Захват
        self.kuka.close_grip()
        time.sleep(1)
        
        # Подъем
        self.kuka.lin_continuous(self.kuka, np.array([approach_point]))
        
        # Перемещение к месту размещения
        self.kuka.lin_continuous(self.kuka, np.array([approach_place]))
        self.kuka.lin_continuous(self.kuka, np.array([place_point]))
        
        # Освобождение
        self.kuka.open_grip()
        time.sleep(1)
        
        # Возврат в исходную позицию
        self.kuka.lin_continuous(self.kuka, np.array([approach_place]))
        
        print("Операция завершена")
    
    def run(self):
        """Основной цикл работы"""
        
        # Место размещения (задается в координатах робота)
        place_position = (400, 300)
        
        try:
            while True:
                # Детектирование объектов
                objects, frame = self.detect_objects(target_color='red')
                
                if objects:
                    # Берем первый обнаруженный объект
                    obj = objects[0]
                    x, y = obj['robot_coords']
                    
                    # Отображение информации
                    cv2.putText(frame, f"Robot coords: ({x:.1f}, {y:.1f})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Press 'p' to pick", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    
                    # Отрисовка bounding box
                    x1, y1, w, h = cv2.boundingRect(obj['contour'])
                    cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
                    cv2.circle(frame, obj['pixel_center'], 5, (0, 0, 255), -1)
                
                cv2.imshow('Color Detection Robot Control', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p') and objects:
                    # Выполнение захвата и перемещения
                    x, y = objects[0]['robot_coords']
                    self.pick_and_place((x, y), place_position)
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.kuka.quit()

if __name__ == "__main__":
    robot = ColorDetectionRobot(robot_ip='192.168.17.2')
    robot.run()

