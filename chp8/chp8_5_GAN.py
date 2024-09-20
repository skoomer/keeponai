# 8 .5 . Введение в генеративно-состязательные сети
# 
# Генеративно-состязательные сети (Generative Adversarial Networks, GAN),


# 8 .5 .1 . Реализация простейшей генеративно- состязательной сети

# Данная реализация — глубокая сверточная генеративно-состязательная сеть (Deep Convolutional GAN, DCGAN), в которой генератор и дискриминатор яв- ляются глубокими сверточными сетями. В ней, например, используется слой Conv2DTranspose для увеличения разрешения изображения в генераторе.

# Мы будем обучать GAN на изображениях из набора CIFAR10, содержащего 50 000 изображений 32 × 32 в формате RGB, которые делятся на 10 классов (по 5000 изображений в каждом классе). Для простоты мы используем только изо- бражения, принадлежащие классу «лягушка».



# В общих чертах GAN выглядит примерно так:
# 1. Сеть generator отображает векторы с формой (размерность_скрытого_простран-
# ства,) в изображения с формой (32, 32, 3).
# 2. Сеть discriminator отображает изображения с формой (32, 32, 3) в оценку
# вероятности того, что изображение является настоящим.
# 3. Сеть gan объединяет генератор и дискриминатор gan(x) = discrimina- tor(generator(x)). То есть сеть gan отображает скрытое пространство векторов в оценку реализма этих скрытых векторов, декодированных генератором.
# 4. Мы обучим дискриминатор на примерах реальных и искусственных изобра- жений, отмеченных метками «настоящее»/«поддельное», как самую обычную модель классификации изображений.
# 5. Для обучения генератора мы используем градиенты весов генератора в отноше- нии потерь модели gan. То есть на каждом шаге мы будем смещать веса генера- тора в направлении увеличения вероятности классификации дискриминатором изображений, декодированных генератором как «настоящие». Иными словами, мы будем обучать генератор обманывать дискриминатор.

# 8 .5 .2 . Набор хитростей


# В качестве последней функции активации в генераторе мы используем tanh вместо sigmoid, которую часто можно встретить в моделях других типов.

# 8 .5 .3 . Генератор

# Листинг 8.29. Сеть генератора в GAN
import keras
from keras import layers
import numpy as np


latent_dim = 32
height = 32
width = 32
channels = 3


generator_input = keras.Input(shape=(latent_dim,))

x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x) 
generator.summary()



# 8 .5 .4 . Дискриминатор

# Теперь перейдем к модели discriminator, которая принимает на входе изображе- ние-кандидат (реальное или искусственное) и относит его к одному из двух клас- сов: «подделка» или «настоящее, имеющееся в обучающем наборе»

# Листинг 8.30. Сеть дискриминатора в GAN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay

discriminator_input = layers.Input(shape=(height, width, channels))

x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4,strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4,strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4,strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x) #Уровень прореживания: важная хитрость!
x = layers.Dense(1, activation='sigmoid')(x) # Уровень классификации
discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

# discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
# discriminator_optimizer = keras.optimizers.RMSprop(
#     lr=0.0008,
#     clipvalue=1.0,# Использование градиентной обрезки (по значению) в оптимизаторе
#     decay=1e-8) # Для стабилизации используется затухание скорости обучения

# Используем планировщик для скорости обучения
learning_rate_schedule = ExponentialDecay(
    initial_learning_rate=0.0004, 
    decay_steps=100000, 
    decay_rate=0.96
)

discriminator_optimizer = RMSprop(
    learning_rate=learning_rate_schedule, 
    clipvalue=1.0
)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

learning_rate_schedule_gan = ExponentialDecay(
    initial_learning_rate=0.0008, 
    decay_steps=100000, 
    decay_rate=0.96
)

# 8 .5 .5 . Состязательная сеть
# Важно также отметить, что дискриминатор нужно «заморозить» на время обучения (отключить его обучение): его веса не должны обновляться при обучении gan. В противном случае все сведется к тому, что вы обучите дискриминатор всегда отвечать «на- стоящее», а это едва ли вам нужно!

# Листинг 8.31. Состязательная сеть

# «Заморозка» весов дискриминатора (это относится только к модели gan)

discriminator.trainable = False



gan_input = keras.Input(shape=(latent_dim,)) 
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan_optimizer = RMSprop(
    learning_rate=learning_rate_schedule_gan,
    clipvalue=1.0
)
# gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')


# 8 .5 .6 . Как обучить сеть DCGAN

# Теперь можно приступать к обучению. Ниже схематически описывается общий цикл обучения. В каждой эпохе нужно выполнить следующие действия:
# 1. Извлечь случайные точки из скрытого пространства (случайный шум).
# 2. Создать изображения с помощью генератора, использовав случайный шум.
# 3. Смешать сгенерированные изображения с настоящими.
# 4. Обучить дискриминатор на этом смешанном наборе изображений, добавив соот- ветствующие цели: «настоящее» (для настоящих изображений) или «подделка» (для сгенерированных изображений).
# 5. Выбрать новые случайные точки из скрытого пространства.
# 6. Обучить gan, использовав эти случайные векторы, с целями, которые всег- да говорят: «это настоящие изображения». Это приведет к смещению весов генератора (и только генератора, потому что внутри gan дискриминатор «за- мораживается») в направлении, увеличивающем вероятность получить от дис- криминатора ответ «настоящее» для сгенерированных изображений: это научит генератор обманывать дискриминатор.



# Листинг 8.32. Реализация обучения GAN

import os
from keras.preprocessing import image

(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data() # load data CIFAR10

# Выбирает изображения лягушек (класс 6)

x_train = x_train[y_train.flatten() == 6]
# Нормализация данных
x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.


iterations = 10000 
batch_size = 20 
save_dir = '/Users/ivan/Vscodebprojects/AIStudy/chp8/gen_img'


start = 0
for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    generated_images = generator.predict(random_latent_vectors)
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)
    d_loss = discriminator.train_on_batch(combined_images, labels)
    # Обучение дискриминатора
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    misleading_targets = np.zeros((batch_size, 1))
    a_loss = gan.train_on_batch(random_latent_vectors,
                            misleading_targets)
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    if step % 100 == 0:
        gan.save_weights('gan.h5')
        print('discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir,'generated_frog' + str(step) + '.png'))
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir,'real_frog' + str(step) + '.png'))
        