import cv2
import numpy as np

cap = cv2.VideoCapture(0) # Указать номер камеры
if cap.isOpened() is True:
    i = 0
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        
        if i % 50 == 0:
            cv2.imwrite(f'calib_{i//50}.png', gray)
        
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (i == 500):
            break
        i += 1

cap.release()
cv2.destroyAllWindows()
