from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator




conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3))

# Выделение признаков с использованием предварительно обученной сверточной основы

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = '/cats_and_dogs_small'
base_dir = '/cats_and_dogs_small_2.h5'
base_dir = os.path.expanduser('~/cats_and_dogs_small')

 
# train_dir = os.path.join(base_dir, 'train')
train_dir = '/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small/train/'

validation_dir = '/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small/validation/'
test_dir = '/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small/test/'
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count,4,4,512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory, 
        target_size=(150, 150),
        batch_size=batch_size, 
        class_mode='binary')
    i = 0 
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels
    
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000) 
    
    
    
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
    
    
    
    # тинг 5.18. Определение и обучение полносвязного классификатора
    
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5), loss='binary_crossentropy',
metrics=['acc'])


history = model.fit(train_features, train_labels, epochs=30,
batch_size=20,
validation_data=(validation_features, validation_labels))



import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc') 
plt.plot(epochs, val_acc, 'b', label='Validation acc') 
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


Листинг 5.21. Полное обучение модели с замороженной сверточной основой


from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers


train_datagen = ImageDataGenerator(
    rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2, 
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
train_dir, #Целевой каталог
target_size=(150, 150), #Приведение всех изображений к размеру 150 × 150 batch_size=20,
class_mode='binary')

validation_generator = test_datagen.flow_from_directory( validation_dir,
target_size=(150, 150), batch_size=20, class_mode='binary')


model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5),
metrics=['acc'])

history = model.fit_generator( train_generator,
steps_per_epoch=100,
epochs=30, validation_data=validation_generator, validation_steps=50)



# до обучение 

# 3амораживание всех слоев, кроме заданных
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
        
model.compile(
    loss='binary_crossentropy', 
    optimizer=optimizers.RMSprop(lr=1e-5),
metrics=['acc'])

history = model.fit_generator(
    train_generator,
steps_per_epoch=100,
epochs=100, 
validation_data=validation_generator,
validation_steps=50)


Построим графики с результатами, использовав тот же код, что и прежде (рис. 5.20
 
 
 Похоже, что кривые искажены помехами. Чтобы можно было уловить тенденцию, сгладим кривые, заменив фактические значения потерь и точности экспоненци- альным скользящим средним. Вот простая вспомогательная функция для этого (рис. 5.22 и 5.23).
 
 
 Теперь наконец можно оценить модель на контрольных данных:
test_generator = test_datagen.flow_from_directory( test_dir,
target_size=(150, 150), batch_size=20, class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50) print('test acc:', test_acc)


5 .3 .3 . Подведение итогов
Вот какие выводы вы должны сделать из примеров, представленных в двух пре- дыдущих разделах:
Сверточные нейронные сети — лучший тип моделей машинного обучения для задач распознавания образов. Вполне можно обучить такую сеть с нуля на очень небольшом наборе данных и получить приличный результат.
Когда объем данных ограничен, главной проблемой становится переобучение. Расширение данных — эффективное средство борьбы с переобучением при работе с изображениями.
Существующую сверточную нейронную сеть с легкостью можно повторно ис- пользовать на новом наборе данных, применив прием выделения признаков. Этот прием особенно ценен при работе с небольшими наборами изображений.
В дополнение к выделению признаков можно использовать прием дообуче- ния, который адаптирует к новой задаче некоторые из представлений, ранее
полученных существующей моделью. Он еще больше повышает качество
модели.
Теперь у вас имеется надежный набор инструментов для решения задач классифи- кации изображений, особенно с ограниченным объемом данных.


# 5 .4 .1 . Визуализация промежуточных активаций
from keras.models import load_model
loaded=tf.keras.models.load_model('cats_and_dogs_small_2.h5')

model = load_model('/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small_2.h5')
model.summary() <1> As a reminder

img_path = '/Users/ivan/Vscodebprojects/AIStudy/kek.jpg'
from keras.preprocessing import image  #Преобразование изображения в четырехмерный тензор
import numpy as np



img = image.load_img(img_path, target_size=(150, 150))

img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

# Модель обучалась на входных данных, которые предварительно были обработаны таким способом

# Отображение тестового изображения
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()



# Листинг 5.27. Создание экземпляра модели из входного тензора и списка выходных тензоров


from keras import models
# Извлечение вывода верхних восьми слоев

layer_outputs = [layer.output for layer in model.layers[:8]] 
layer_outputs = [layer.output for layer in loaded.layers[:8]] 

activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activation_model = Model(inputs=model.input, outputs=layer_outputs)




Листинг 5.28. Запуск модели в режиме прогнозирования

activations = activation_model.predict(img_tensor)



Возьмем для примера активацию первого сверточного слоя для входного изобра- жения кошки:
first_layer_activation = activations[0]

print(first_layer_activation.shape)


Листинг 5.29. Визуализация четвертого канала 

import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0, :, :, 10], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 11], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 12], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 13], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 14], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 15], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 16], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 17], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 18], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 19], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 20], cmap='viridis')

# Листинг 5.31. Визуализация всех каналов для всех промежуточных активаций



 
layer_names = [ ]

for layer in model.layers[:8]:
    #Извлечь имена слоев для отображения
    # на рисунке
    layer_names.append(layer.name)
    
images_per_row = 8

for layer_name , layer_activation in zip(layer_names, activations ): # cикл отображения карт признаков
    n_features = layer_activation.shape[-1] # kоличество признаков в карте признаков
    size = layer_activation.shape[1] # # Карта признаков имеет форму # (1, size, size, n_features)
                                        
    n_cols = n_features // images_per_row # Количество колонок в матрице отображения каналов
    
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):#Вывод каждого фильтра в большую горизонтальную сетку
        for row in range(images_per_row):
            channel_image = layer_activation[0,:, :, col * images_per_row + row]
            channel_image -= channel_image.mean()  # Заключительная обработка признака, чтобы получить приемлемую визуализацию
            channel_image /= channel_image.std() 
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, #Вывод сетки
                         row * size : (row + 1) * size] = channel_image
scale = 1. / size

plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))

plt.title(layer_name)
plt.grid(False)
plt.imshow(display_grid, aspect='auto', cmap='viridis')


layer_names = [layer.name for layer in model.layers[:12]]


for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean() 
            if channel_image.std() > 0:
                channel_image /= (channel_image.std() + 1e-5)
            
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
            
            

5 .4 .3 . Визуализация тепловых карт активации класса

В этом разделе описывается еще один прием визуализации, позволяющий понять, какие части данного изображения помогли сверточной нейронной сети принять окончательное решение о его классификации. Это полезно для отладки процесса принятия решений в сверточной нейронной сети, особенно в случае ошибок клас- сификации. Он также помогает определить местоположение конкретных объектов на изображении.

Категория методов, описываемых здесь, называется визуализацией карты акти- вации класса (Class Activation Map, CAM)