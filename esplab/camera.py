# import upip
# upip.install('micropython-picoweb')
# upip.install('micropython-ulogging')



# import esp32
# from machine import Pin

# # Настройки пинов для камеры
# CAMERA_PINS = {
#     'pwdn': 32,
#     'reset': -1,   # Не используется
#     'xclk': 0,
#     'sioc': 27,
#     'siod': 26,
#     'y9': 35,
#     'y8': 34,
#     'y7': 39,
#     'y6': 36,
#     'y5': 21,
#     'y4': 19,
#     'y3': 18,
#     'y2': 5,
#     'vsync': 25,
#     'href': 23,
#     'pclk': 22,
# }

# def init_camera():
#     esp32.camera_init(
#         pwdn=CAMERA_PINS['pwdn'],
#         reset=CAMERA_PINS['reset'],
#         xclk=CAMERA_PINS['xclk'],
#         sioc=CAMERA_PINS['sioc'],
#         siod=CAMERA_PINS['siod'],
#         y9=CAMERA_PINS['y9'],
#         y8=CAMERA_PINS['y8'],
#         y7=CAMERA_PINS['y7'],
#         y6=CAMERA_PINS['y6'],
#         y5=CAMERA_PINS['y5'],
#         y4=CAMERA_PINS['y4'],
#         y3=CAMERA_PINS['y3'],
#         y2=CAMERA_PINS['y2'],
#         vsync=CAMERA_PINS['vsync'],
#         href=CAMERA_PINS['href'],
#         pclk=CAMERA_PINS['pclk']
#     )

# def capture_photo():
#     buf = esp32.camera_capture()
#     return buf



# main.py


# import network
# import picoweb
# from camera import init_camera, capture_photo
# from machine import Pin, SPI, SDCard

# # Инициализация камеры
# init_camera()

# # Подключение к Wi-Fi
# SSID = 'your_SSID'
# PASSWORD = 'your_PASSWORD'

# station = network.WLAN(network.STA_IF)
# station.active(True)
# station.connect(SSID, PASSWORD)

# while not station.isconnected():
#     pass

# print('Connection successful')
# print(station.ifconfig())

# # Настройка SD карты (если используется)
# sd = SDCard(slot=2, sck=14, mosi=15, miso=2, cs=13)
# uos.mount(sd, '/sd')

# # Веб-сервер
# app = picoweb.WebApp('camera_app')

# @app.route('/')
# def index(req, resp):
#     yield from picoweb.start_response(resp, content_type='text/html')
#     yield from resp.awrite("""
#         <html>
#             <body>
#                 <h1>ESP32-CAM</h1>
#                 <img src="/photo" width="640" height="480">
#                 <form action="/photo" method="post">
#                     <input type="submit" value="Capture Photo">
#                 </form>
#             </body>
#         </html>
#     """)

# @app.route('/photo')
# def photo(req, resp):
#     yield from picoweb.start_response(resp, content_type='image/jpeg')
#     buf = capture_photo()
#     yield from resp.awrite(buf)

# # Запуск веб-сервера
# app.run(debug=True, host='0.0.0.0', port=80)




