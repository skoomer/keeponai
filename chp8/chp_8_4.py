
# 8 .4 . Генерирование изображений
# с вариационными автокодировщиками

# Используя GAN и VAE, можно создавать скрытые пространства звуков, музыки или даже текста, однако на практике наиболее инте- ресные результаты получаются с изображениями, и именно поэтому мы сосредо- точимся на этом направлении.

# 8 .4 .2 . Концептуальные векторы для редактирования изображений

# 8 .4 .3 . Вариационные автокодировщики

# Давайте пройдемся по реализации вариационного автокодировщика в Keras. Схематически она вы- глядит так:
    
z_mean, z_log_variance = encoder(input_img) #Кодирование входа в среднее и дисперсию

z = z_mean + exp(z_log_variance) * epsilon # извлечение скрытой точки с использв случайной величины  epsilon

reconstructed_img = decoder(z) # Декодирование z обратно в изображение

model = Model(input_img, reconstructed_img) #Создание модели автокодировщика кот отобр вход изобр  в его констр



# В следующем листинге демонстрируется используемая нами сеть кодировщика, отображающая изображения в параметры распределения вероятности в скрытом пространстве. Эта простая сверточная сеть отображает входное изображение x в два вектора, z_mean и z_log_var.



# Листинг 8.23. Сеть кодировщика VAE

import keras
from keras import layers
from keras import backend as K
from keras.models import Model 
import numpy as np


img_shape = (28, 28, 1) batch_size = 16

latent_dim = 2  # Размерность скрытого пространства: двумерная плоскость

input_img = keras.Input(shape=img_shape)
x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img) 

x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)


z_mean = layers.Dense(latent_dim)(x) #Входное изображение кодируется
z_log_var = layers.Dense(latent_dim)(x) #в эти два параметра

# В Keras все сущее должно заключаться в слои, поэтому код, не являющийся частью встроенного слоя, необходимо заключать в слой Lambda (или в свой собственный слой).

# Листинг 8.24. Функция выбора точки из скрытого пространства

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
    mean=0., stddev=1.) 
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# Листинг 8.25. Сеть декодера VAE, отображающая точки из скрытого пространства в изображения

decoder_input = layers.Input(K.int_shape(z)[1:]) # Передача z на вход
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input) # увеличение разрешения входа


# Использование слоев Conv2DTranspose и Conv2D для декодирования z в карту признаков с тем же размером, что и входное изображение

x = layers.Reshape(shape_before_flattening[1:])(x) # Преобразование z в карту признаков с той же формой, которую имела карта признаков перед последним слоем Flatten в модели кодировщика

x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x) #

x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x) # 

decoder = Model(decoder_input, x) #Создается модель декодера, которая преобразует "decoder_input" в декодированное изображение

z_decoded = decoder(z) #Применяет декодер к z, чтобы восстановить декодированное значение z


# Двойственная природа потерь VAE не соответствует традиционной форме loss(input, target). Поэтому мы реализуем вычисление потерь, написав свой слой, который внутренне использует встроенный метод add_loss слоя для создания произвольных потерь.


# Листинг 8.26. Собственный слой для вычисления потерь VAE 

class CustomVariationalLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)
    
    def call(self, inputs): x = inputs[0]
        # Собственные слои реализуются определением метода call
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded) 
        self.add_loss(loss, inputs=inputs)
        #Мы не используем этот результат, однако слой должен что-то возвращать
        return x
    
#Вызов собственного слоя с исходными и декодированными данными для получения окончательного вывода модели
y = CustomVariationalLayer()([input_img, z_decoded])

# Наконец, мы готовы создать и обучить модель. Поскольку вычислением потерь у нас занимается собственный слой, мы не указываем функцию потерь на этапе компиляции (loss=None). Это, в свою очередь, означает, что нам не нужно пере- давать целевые данные в процесс обучения (как можно заметить, в метод fit обу- чаемой модели передается только x_train).

# Листинг 8.27. Обучение VAE

from keras.datasets import mnist
vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()


(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255. 
x_train = x_train.reshape(x_train.shape + (1,)) 
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape + (1,))


vae.fit(
    x=x_train,
    y=None, 
    shuffle=True
    ,epochs=10, 
    batch_size=batch_size, 
    validation_data=(x_test, None)
    )

# После обучения такой модели — в данном случае на наборе MNIST — мы можем использовать сеть decoder для превращения произвольных векторов из скрытого пространства в изображения


# Листинг 8.28. Выбор сетки с точками из двумерного скрытого пространства и их декодирование в изображения

import matplotlib.pyplot as plt
from scipy.stats import norm


#Будет отображаться сетка 15 × 15 цифр (всего 255 цифр)
n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
#Преобразует координаты линей- ного пространства с использова- нием функции ppf из пакета SciPy для получения значений скрытой переменной z (поскольку пред- шествующее скрытое простран- ство является гауссовым)
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        # Многократное повторение выбора z для формирования полного пакета
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
    #Декодирование пакета в изображения цифр
    x_decoded = decoder.predict(z_sample, batch_size=batch_size)
    #Преобразование первой цифры в пакете из размерности 28 × 28 × 1 в 28 × 28
    digit = x_decoded[0].reshape(digit_size, digit_size)
    figure[i * digit_size: (i + 1) * digit_size,
           j * digit_size: (j + 1) * digit_size] = digit
    
    
plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()


Сетка выбранных цифр (рис. 8.14) демонстрирует полностью непрерывное распре- деление разных классов цифр, где одна цифра превращается в другую по пути через непрерывное скрытое пространство. Конкретные направления в этом пространстве наделены определенным смыслом: например, есть направление «четверочности», «единичности» и т. д.

8 .4 .4 . Подведение итогов
Большинство успешных практических применений в области графики, которые мне приходилось видеть, основаны на вариационных автокодировщиках, а гене- ративно-состязательные сети пользуются очень большой популярностью в акаде- мической среде, по крайней мере так было в 2016–2017 годах. Как они действуют и как реализуются, вы узнаете в следующем разделе.

СОВЕТ
Для экспериментов с генерацией изображений я рекомендую использовать набор Largescale Celeb Faces Attributes (CelebA). Этот набор доступен для загрузки бес- платно и содержит более 200 000 портретов знаменитостей. В частности, он пре- красно подходит для экспериментов с концептуальными векторами и в этом смысле превосходит набор MNIST.