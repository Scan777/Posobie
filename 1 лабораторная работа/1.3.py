# Калибровка камеры
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Сохранение параметров
np.savez('calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Вывод результатов
print("Матрица камеры:")
print(mtx)
print("\nКоэффициенты искажения:")
print(dist)
