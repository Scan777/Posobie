import pyrealsense2 as rs
import cv2
import numpy as np

class RealsenseCamera:
    """Упрощенная версия класса для работы с камерой RealSense"""
    
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)
        
        # Получение сенсора глубины для настройки
        sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        self.depth_scale = sensor.get_depth_scale()
        print(f"Depth scale: {self.depth_scale}")
    
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image, depth_frame
    
    def get_distance(self, depth_frame, x, y):
        """Получение расстояния до точки с координатами (x, y) в метрах"""
        distance = depth_frame.get_distance(x, y)
        return distance
    
    def release(self):
        self.pipeline.stop()

# Основная программа
def main():
    # Инициализация камеры
    camera = RealsenseCamera()
    
    # Переменные для хранения координат мыши
    mouse_x, mouse_y = -1, -1
    mouse_clicked = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y, mouse_clicked
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_x, mouse_y = x, y
        elif event == cv2.EVENT_LBUTTONDOWN:
            mouse_clicked = True
            mouse_x, mouse_y = x, y
    
    cv2.namedWindow('RealSense Test')
    cv2.setMouseCallback('RealSense Test', mouse_callback)
    
    try:
        while True:
            # Получение кадров
            color_image, depth_image, depth_frame = camera.get_frame()
            
            if color_image is None:
                continue
            
            # Если мышь перемещается, показываем расстояние
            if mouse_x >= 0 and mouse_y >= 0:
                # Проверка границ
                if mouse_x < color_image.shape[1] and mouse_y < color_image.shape[0]:
                    # Получение расстояния
                    distance = camera.get_distance(depth_frame, mouse_x, mouse_y)
                    
                    # Отрисовка точки и расстояния
                    cv2.circle(color_image, (mouse_x, mouse_y), 5, (0, 0, 255), -1)
                    
                    if distance > 0:
                        text = f"Distance: {distance:.3f} m ({distance*1000:.1f} mm)"
                        cv2.putText(color_image, text, 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(color_image, "No depth data", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 0, 255), 2)
                    
                    if mouse_clicked:
                        print(f"Clicked at ({mouse_x}, {mouse_y}): {distance*1000:.1f} mm")
                        mouse_clicked = False
            
            # Создание цветной карты глубины для визуализации
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Объединение изображений для отображения
            images = np.hstack((color_image, depth_colormap))
            
            cv2.imshow('RealSense Test', images)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
