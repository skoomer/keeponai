from machine import Pin
import network
import socket
import gc
import time
import esp32
from esp32 import Camera

# Настройка камеры
Camera.init(0)

# Настройка Wi-Fi
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect('SSID', 'PASSWORD')  # Замените на свои SSID и пароль
while not wlan.isconnected():
    pass

# Создание веб-сервера
addr = socket.getaddrinfo('0.0.0.0', 80)[0][-1]
s = socket.socket()
s.bind(addr)
s.listen(1)

while True:
    cl, addr = s.accept()
    print('Client connected from', addr)
    request = cl.recv(1024)
    cl.send('HTTP/1.1 200 OK\r\n')
    cl.send('Content-Type: multipart/x-mixed-replace; boundary=frame\r\n')
    cl.send('\r\n')

    while True:
        img = Camera.capture()
        if img:
            cl.send(b'--frame\r\n')
            cl.send(b'Content-Type: image/jpeg\r\n')
            cl.send(b'Content-Length: %d\r\n' % len(img))
            cl.send(b'\r\n')
            cl.send(img)
            cl.send(b'\r\n')
        else:
            break

    cl.close()
Установка Python и необходимых библиотек:

На компьютере, где ты будешь просматривать поток, установи Python и необходимые библиотеки, например requests для HTTP запросов или OpenCV для обработки видео.
Подключение к веб-камере:

Открой браузер и перейди по IP-адресу твоего ESP32-CAM, чтобы увидеть потоковое видео. IP-адрес будет отображён в логах твоего скрипта на ESP32.
Если у тебя возникнут вопросы или проблемы на каком-то этапе, дай знать!