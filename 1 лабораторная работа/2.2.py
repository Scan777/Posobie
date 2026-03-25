# Преобразование в HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)

fig = plt.figure(figsize=(8,6))
axis = fig.add_subplot(111, projection="3d")

axis.scatter(v.flatten(), h.flatten(), s.flatten(), 
             facecolors=pixel_colors, marker='.')
axis.set_xlabel("Value")
axis.set_ylabel("Hue")
axis.set_zlabel("Saturation")
plt.title("HSV цветовое пространство")
plt.show()
