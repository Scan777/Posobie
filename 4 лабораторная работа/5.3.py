def calibrate_camera_robot(pixel_points, robot_points):
    """
    Калибровка преобразования пиксельных координат в координаты робота
    
    pixel_points: массив точек в пикселях (N×2)
    robot_points: массив точек в координатах робота (N×2)
    """
    # Решение системы линейных уравнений
    A = np.zeros((2*len(pixel_points), 6))
    b = np.zeros((2*len(pixel_points), 1))
    
    for i in range(len(pixel_points)):
        x_p, y_p = pixel_points[i]
        x_r, y_r = robot_points[i]
        
        A[2*i, :] = [x_p, y_p, 1, 0, 0, 0]
        A[2*i+1, :] = [0, 0, 0, x_p, y_p, 1]
        
        b[2*i] = x_r
        b[2*i+1] = y_r
    
    # Решение
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    return params
