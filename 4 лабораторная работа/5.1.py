from openshowvar import openshowvar
from Kuka import Kuka
import numpy as np
import time

# Подключение к роботу
robot = openshowvar(ip='192.168.17.2', port=7000)
kuka = Kuka(robot)

# Чтение текущего положения
kuka.read_cartesian()
print(f"Текущее положение: X={kuka.x_cartesian}, Y={kuka.y_cartesian}, Z={kuka.z_cartesian}")

# Простое движение
test_point = np.array([300, 200, 100, -180, 0, 0])
kuka.set_base(1)
kuka.set_tool(1)
kuka.set_speed(30)
kuka.lin_continuous(kuka, np.array([test_point]))

time.sleep(2)
kuka.quit()
