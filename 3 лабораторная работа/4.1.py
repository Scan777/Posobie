import pyrealsense2 as rs
import cv2
import numpy as np

class DepthObjectDetector:
    """Класс для детектирования объектов на основе глубины"""
    
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)
        
        # Получение сенсора глубины
        sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        self.depth_scale = sensor.get_depth_scale()
        print(f"Depth scale: {self.depth_scale}")
    
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None, None
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image, depth_frame
    
    def get_depth_mask(self, depth_image, min_distance, max_distance):
        """
        Создание маски для пикселей, попадающих в диапазон расстояний
        min_distance, max_distance - в миллиметрах
        """
        # Преобразование расстояний в метры
        min_dist_m = min_distance / 1000.0
        max_dist_m = max_distance / 1000.0
        
        # Создание маски
        mask = cv2.inRange(depth_image, min_dist_m / self.depth_scale, 
                          max_dist_m / self.depth_scale)
        
        # Морфологические операции для удаления шума
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def find_objects(self, mask, min_area=500):
        """Поиск контуров объектов на маске"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                # Вычисление ограничивающей рамки
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Вычисление центра масс
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = x + w//2, y + h//2
                
                objects.append({
                    'contour': cnt,
                    'bbox': (x, y, w, h),
                    'center': (cx, cy),
                    'area': area
                })
        
        return objects
    
    def release(self):
        self.pipeline.stop()

def main():
    # Инициализация детектора
    detector = DepthObjectDetector()
    
    # Параметры детектирования
    MIN_DISTANCE = 500  # мм
    MAX_DISTANCE = 800  # мм
    MIN_AREA = 500      # пикселей
    
    try:
        while True:
            # Получение кадров
            color_image, depth_image, depth_frame = detector.get_frame()
            
            if color_image is None:
                continue
            
            # Создание маски по глубине
            depth_mask = detector.get_depth_mask(depth_image, MIN_DISTANCE, MAX_DISTANCE)
            
            # Поиск объектов
            objects = detector.find_objects(depth_mask, MIN_AREA)
            
            # Визуализация результатов
            result_image = color_image.copy()
            
            for obj in objects:
                x, y, w, h = obj['bbox']
                cx, cy = obj['center']
                
                # Отрисовка bounding box
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Отрисовка центра
                cv2.circle(result_image, (cx, cy), 5, (0, 0, 255), -1)
                
                # Получение расстояния до центра объекта
                distance = depth_frame.get_distance(cx, cy) * 1000  # в мм
                
                # Вывод информации
                cv2.putText(result_image, f"Dist: {distance:.1f} mm", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 0), 1)
                cv2.putText(result_image, f"Area: {obj['area']}", 
                           (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 0), 1)
            
            # Информация о количестве объектов
            cv2.putText(result_image, f"Objects: {len(objects)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2)
            cv2.putText(result_image, f"Range: {MIN_DISTANCE}-{MAX_DISTANCE} mm", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2)
            
            # Создание цветной карты глубины
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Объединение изображений
            top_row = np.hstack((result_image, depth_colormap))
            bottom_row = cv2.cvtColor(depth_mask, cv2.COLOR_GRAY2BGR)
            bottom_row = cv2.resize(bottom_row, (top_row.shape[1], bottom_row.shape[0]))
            display = np.vstack((top_row, bottom_row))
            
            cv2.imshow('Depth-based Object Detection', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                MIN_DISTANCE += 50
                print(f"Range: {MIN_DISTANCE}-{MAX_DISTANCE} mm")
            elif key == ord('-'):
                MIN_DISTANCE = max(0, MIN_DISTANCE - 50)
                print(f"Range: {MIN_DISTANCE}-{MAX_DISTANCE} mm")
    
    finally:
        detector.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
