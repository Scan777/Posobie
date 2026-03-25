import pyrealsense2 as rs
import numpy as np
import cv2

# Создание конвейера
pipeline = rs.pipeline()

# Настройка конфигурации
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Запуск конвейера
pipeline.start(config)

try:
    while True:
        # Ожидание нового кадра
        frames = pipeline.wait_for_frames()
        
        # Получение кадров цвета и глубины
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue
        
        # Преобразование в массивы numpy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Нормализация глубины для визуализации
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # Отображение результатов
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Map', depth_colormap)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

