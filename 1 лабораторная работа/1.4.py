# Загрузка параметров
with np.load('calib.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# Коррекция изображения
img = cv.imread('test_image.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# Обрезка изображения
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
