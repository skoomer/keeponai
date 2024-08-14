import network
import time
import camera
from MicroWebSrv import microWebSrv   # Import the web server module

import uasyncio as asyncio

# Wi-Fi credentials
ssid_ = "skoomer"
wp2_pass = "mariklol1245"

# Connect to Wi-Fi
sta_if = network.WLAN(network.STA_IF)
sta_if.active(True)
sta_if.connect(ssid_, wp2_pass)

# Wait for connection
for i in range(10):
    if not sta_if.isconnected():
        time.sleep(1)
    else:
        break

if sta_if.isconnected():
    print("Connected to Wi-Fi")
    print("IP address:", sta_if.ifconfig()[0])
else:
    print("Failed to connect to Wi-Fi")
    sta_if.active(False)

# Initialize camera
def init_camera():
    try:
        camera.init(0, format=camera.JPEG)
        print("Camera initialized successfully")
    except Exception as e:
        print("Camera Init Failed:", e)

# Call the camera initialization function
init_camera()
file_counter = 1



# Define the handler for taking photos
def _httpHandlerTakePhoto(httpClient, httpResponse):
    global file_counter
    buffer = camera.capture()  # Capture photo

    file_path = f"/captured_image_{file_counter}.jpg"
    with open(file_path, "wb") as file:  # Use binary mode for writing the image
        file.write(buffer)

    file_counter += 1
    print(f"Photo saved as {file_path}")

    # Send the photo as a response
    httpResponse.WriteResponseFile(file_path, contentType="image/jpeg")

srv = microWebSrv.MicroWebSrv()  # Инициализация сервера

# Define the routes for the web server
srv = microWebSrv.MicroWebSrv(routeHandlers=[
    ("/photo", "GET", _httpHandlerTakePhoto)
])

# Start the web server
srv.Start(threaded=True)
print("Web server started!")

# Функция очистки
def cleanup():
    print("Отключение камеры и Wi-Fi")
    camera.deinit()
    sta_if.disconnect()

# Выполнение функции очистки при завершении программы
try:
    while True:
        time.sleep(1)  # Поддержка работы сервера
except KeyboardInterrupt:
    pass
finally:
    cleanup()


