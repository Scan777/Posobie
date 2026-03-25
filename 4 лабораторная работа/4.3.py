import cv2
import numpy as np
import pyrealsense2 as rs
import time
from Kuka import Kuka
from openshowvar import openshowvar

class StereoVisionRobotControl:
    """Класс для управления роботом на основе стереозрения"""
    
    def __init__(self, robot_ip='192.168.1.2', port=7000):
        # Подключение к роботу
        self.robot = openshowvar(ip=robot_ip, port=port)
        self.kuka = Kuka(self.robot)
        
        # Инициализация стереокамеры Intel RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)
        
        # Получение параметров камеры
        self.profile = self.pipeline.get_active_profile()
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        
        # Получение внутренних параметров
        self.color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = self.color_stream.as_video_stream_profile().get_intrinsics()
        
        print(f"Depth scale: {self.depth_scale}")
        print(f"Camera intrinsics: fx={self.intrinsics.fx}, fy={self.intrinsics.fy}")
        
        # Параметры рабочей зоны
        self.robot_base = {
            'x': 200,  # смещение робота относительно камеры
            'y': 150,
            'z': 0
        }
        
        self.workspace = {
            'z_pick': 50,      # высота захвата
            'z_approach': 100, # высота подхода
            'min_distance': 300,  # минимальное рабочее расстояние (мм)
            'max_distance': 800   # максимальное рабочее расстояние (мм)
        }
    
    def get_frames(self):
        """Получение кадров цвета и глубины"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None, None
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image, depth_frame
    
    def depth_to_robot_coords(self, x_pixel, y_pixel, depth_frame):
        """Преобразование пиксельных координат с глубиной в координаты робота"""
        
        # Получение глубины в точке
        depth = depth_frame.get_distance(x_pixel, y_pixel) * 1000  # в мм
        
        if depth == 0 or depth < self.workspace['min_distance'] or depth > self.workspace['max_distance']:
            return None, None, None
        
        # Преобразование в координаты камеры
        x_cam = (x_pixel - self.intrinsics.ppx) * depth / self.intrinsics.fx
        y_cam = (y_pixel - self.intrinsics.ppy) * depth / self.intrinsics.fy
        z_cam = depth
        
        # Преобразование в координаты робота
        x_robot = x_cam + self.robot_base['x']
        y_robot = y_cam + self.robot_base['y']
        z_robot = z_cam + self.robot_base['z']
        
        return x_robot, y_robot, z_robot
    
    def detect_objects_by_depth(self, depth_image, depth_frame):
        """Обнаружение объектов по глубине"""
        
        # Создание маски для рабочего диапазона глубин
        min_dist = self.workspace['min_distance'] / 1000.0
        max_dist = self.workspace['max_distance'] / 1000.0
        
        depth_mask = cv2.inRange(depth_image, min_dist / self.depth_scale, 
                                 max_dist / self.depth_scale)
        
        # Морфологические операции
        kernel = np.ones((5,5), np.uint8)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)
        
        # Поиск контуров
        contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # минимальная площадь
                # Вычисление центра
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Получение координат робота
                    x_robot, y_robot, z_robot = self.depth_to_robot_coords(cx, cy, depth_frame)
                    
                    if x_robot is not None:
                        objects.append({
                            'center': (cx, cy),
                            'robot_coords': (x_robot, y_robot, z_robot),
                            'area': area,
                            'contour': cnt
                        })
        
        return objects
    
    def pick_object(self, robot_coords):
        """Захват объекта по трехмерным координатам"""
        
        x, y, z = robot_coords
        
        print(f"Захват объекта: X={x:.1f}, Y={y:.1f}, Z={z:.1f} мм")
        
        # Установка параметров
        self.kuka.set_base(1)
        self.kuka.set_tool(1)
        self.kuka.set_speed(30)
        
        # Точка подхода (над объектом)
        approach_point = np.array([
            x, y,
            z + self.workspace['z_approach'],
            -180, 0, 0
        ])
        
        # Точка захвата
        pick_point = np.array([
            x, y,
            z + self.workspace['z_pick'],
            -180, 0, 0
        ])
        
        # Движение к объекту
        self.kuka.lin_continuous(self.kuka, np.array([approach_point]))
        self.kuka.lin_continuous(self.kuka, np.array([pick_point]))
        
        # Захват
        self.kuka.close_grip()
        time.sleep(1)
        
        # Подъем
        self.kuka.lin_continuous(self.kuka, np.array([approach_point]))
        
        return True
    
    def run(self):
        """Основной цикл работы"""
        
        try:
            while True:
                # Получение кадров
                color_image, depth_image, depth_frame = self.get_frames()
                
                if color_image is None:
                    continue
                
                # Обнаружение объектов
                objects = self.detect_objects_by_depth(depth_image, depth_frame)
                
                # Создание цветной карты глубины
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # Визуализация
                for obj in objects:
                    cx, cy = obj['center']
                    x_robot, y_robot, z_robot = obj['robot_coords']
                    
                    # Отрисовка на цветном изображении
                    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.drawContours(color_image, [obj['contour']], -1, (0, 255, 0), 2)
                    
                    # Информация о координатах
                    cv2.putText(color_image, 
                               f"Robot: ({x_robot:.0f}, {y_robot:.0f}, {z_robot:.0f})", 
                               (cx - 100, cy - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Общая информация
                cv2.putText(color_image, f"Objects: {len(objects)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                cv2.putText(color_image, f"Depth range: {self.workspace['min_distance']}-{self.workspace['max_distance']} mm", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 2)
                cv2.putText(color_image, "Press 'p' to pick first object", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 2)
                
                # Объединение изображений
                images = np.hstack((color_image, depth_colormap))
                
                cv2.imshow('Stereo Vision Robot Control', images)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p') and objects:
                    # Захват первого объекта
                    self.pick_object(objects[0]['robot_coords'])
        
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            self.kuka.quit()

if __name__ == "__main__":
    controller = StereoVisionRobotControl(robot_ip='192.168.17.2')
    controller.run()
