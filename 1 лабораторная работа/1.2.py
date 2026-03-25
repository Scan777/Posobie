import numpy as np
import cv2 as cv
import glob

# Критерии остановки уточнения углов
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Подготовка точек объекта (например, для доски 8x5 внутренних углов)
objp = np.zeros((8*5, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)

# Массивы для хранения точек
objpoints = []  # 3D точки в реальном мире
imgpoints = []  # 2D точки на изображении

images = glob.glob('*.png')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Поиск углов шахматной доски
    ret, corners = cv.findChessboardCorners(gray, (8,5), None)
    
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        # Отображение углов
        cv.drawChessboardCorners(img, (8,5), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()
