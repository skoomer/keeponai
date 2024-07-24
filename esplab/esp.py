
# /dev/tty.usbserial-14110

# esptool.py --port /dev/tty.usbserial-14110 erase_flash

# esptool.py --chip esp32 --port /dev/tty.usbserial-14110 write_flash -z -fm dio  0x1000 micropython_camera_feeeb5ea3_esp32_idf4_4.bin

import sensor
import image
import time
import os

# Инициализация камеры
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)

# Делаем снимок
img = sensor.snapshot()

# Сохраняем изображение на SD-карту, если она установлена
try:
    os.listdir('/sd')  # Проверяем, доступна ли SD-карта
    img_path = '/sd/photo.jpg'
    img.save(img_path)
    print(f"Фото сохранено на SD-карту: {img_path}")
except OSError:
    print("SD-карта не найдена, сохранение на устройство")

# Добавьте код для отправки изображения по сети (например, по HTTP)

