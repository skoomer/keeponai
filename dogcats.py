import os, shutil
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Путь к каталогу с распакованным исходным набором данных
original_dataset_dir = '/Users/ivan/Vscodebprojects/AIStudy/kaggle_original_data'
original_dataset_dir_train = '/Users/ivan/Vscodebprojects/AIStudy/kaggle_original_data/train'
original_dataset_dir_test1 = '/Users/ivan/Vscodebprojects/AIStudy/kaggle_original_data/test1'

# Каталог для сохранения выделенного небольшого набора
base_dir = '/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small'
# os.mkdir(base_dir)


train_dir = os.path.join(base_dir, 'train') #Каталоги для 
# os.mkdir(train_dir) #обучающего, 

validation_dir = os.path.join(base_dir, 'validation') # проверочного
# os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test') #и конттрольного поднабора 
# os.mkdir(test_dir)


train_cats_dir = os.path.join(train_dir, 'cats') # Каталог для обучающих изображений с кошками
# os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs') # Каталог для обучающих изображений с собаками
# os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats') # Каталог для про- верочных изобра- жений с кошками
# os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs') # Каталог для про- верочных изобра- жений с dogs
# os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats') #Каталог для контрольных изображений с кошками
# os.mkdir(test_cats_dir)

# Каталог для контрольных изображений с собаками
test_dogs_dir = os.path.join(test_dir, 'dogs') # 
# os.mkdir(test_dogs_dir)


# fnames = ['cat.{}.jpg'.format(i) for i in range(1000)] 

# for fname in fnames:
#     #Копирование первых 1000 изображений
#     # с кошками в каталог train_cats_dir
#     src = os.path.join(original_dataset_dir_train, fname) 
#     dst = os.path.join(train_cats_dir, fname) 
#     shutil.copyfile(src, dst)
    
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)] 
# for fname in fnames:
#     # Копирование сле- дующих 500 изо- бражений с кош- ками в каталог validation_cats_dir
#     src = os.path.join(original_dataset_dir_train, fname)
#     dst = os.path.join(validation_cats_dir, fname) 
#     shutil.copyfile(src, dst)
    
# fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)] 
# for fname in fnames:
#     # Копирование сле- дующих 500 изо- бражений с кош- ками в каталог test_cats_dir
#     src = os.path.join(original_dataset_dir_train, fname)
#     dst = os.path.join(test_cats_dir, fname) 
#     shutil.copyfile(src, dst)
    
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
# #     Копирование первых 1000 изображений
# # с собаками в каталог train_dogs_dir
#     src = os.path.join(original_dataset_dir_train, fname)
#     dst = os.path.join(train_dogs_dir, fname)
#     shutil.copyfile(src, dst)
    
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     # Копирование сле- дующих 500 изо- бражений с соба- ками в каталог validation_dogs_dir
#     src = os.path.join(original_dataset_dir_train, fname)
#     dst = os.path.join(validation_dogs_dir, fname) 
#     shutil.copyfile(src, dst)
    
# fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     # Копирование сле- дующих 500 изо- бражений с со- баками в каталог test_dogs_dir
#     src = os.path.join(original_dataset_dir_train, fname)
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dst)
    
# Для проверки подсчитаем, сколько изображений оказалось в каждом поднаборе (обучающем/проверочном/контрольном):
# print('total training cat images:', len(os.listdir(train_cats_dir)))

# print('total training dog images:', len(os.listdir(train_dogs_dir)))

# print('total validation cat images:', len(os.listdir(validation_cats_dir)))

# print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

# print('total test cat images:', len(os.listdir(test_cats_dir))) 
# print('total test dog images:', len(os.listdir(test_dogs_dir))) 


# Итак, у нас действительно имеется 2000 обучающих, 1000 проверочных и 1000 кон- трольных изображений. Каждый поднабор содержит одинаковое количество об- разцов каждого класса: это сбалансированная задача бинарной классификации, соответственно, мерой успеха может служить точность классификации.
# 
# 
# 5 .2 .3 . Конструирование сети

# Листинг 5.5. Создание небольшой сверточной нейронной сети для классификации
# изображений кошек и собак

from keras import layers
from keras import models
from keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = models.Sequential() 

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))



model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# На этапе компиляции, как обычно, используем оптимизатор RMSprop. Так как сеть заканчивается единственным признаком, используем функцию потерь binary_ crossentropy (для напоминания: в табл. 4.1 приводится шпаргалка по использова- нию разных функций потерь в разных ситуациях).

# Листинг 5.6. Настройка модели для обучения


model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4),metrics=['acc'])


# from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1./255) #Масштабировать значения
test_datagen = ImageDataGenerator(rescale=1./255) #с коэффициентом 1/255


train_generator = train_datagen.flow_from_directory(
    train_dir, #Целевой каталог
    target_size=(150, 150), #Привести все изображения к размеру 150 × 150
    batch_size=20,
    class_mode='binary') # Так как используется функция потерь binary_crossentropy, метки должны быть бинарными

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
 target_size=(150, 150),
 batch_size=20,
class_mode='binary')


# def generator():
#     i= 0
#     while True:
#         i += 1
#         yield i
        
# for item in generator():
#        print(item)
#        if item > 4:
#            break


# for data_batch, labels_batch in train_generator:
#     print('data batch shape:', data_batch.shape)
#     print('labels batch shape:', labels_batch.shape)
#     break

# history = model.fit_generator(
#     train_generator,
# steps_per_epoch=100,
# epochs=30,
# validation_data=validation_generator,
# validation_steps=50)



history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('cats_and_dogs_small_1.h5')


# Листинг 5.10. Формирование графиков изменения потерь и точности в процессе обучения
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



# На графиках четко наблюдается эффект переобучения. Точность на обучающих данных линейно растет и приближается к 100 %, тогда как точность на проверочных данных останавливается на отметке 70–72 %. Потери на этапе проверки достигают минимума всего после пяти эпох и затем замирают, а потери на этапе обучения продолжают линейно уменьшаться, почти достигая 0.
# Поскольку у нас относительно немного обучающих образцов (2000), переобу- чение становится проблемой номер один. Вы уже знаете несколько методов, по- могающих смягчить эту проблему, таких как прореживание и сокращение весов

# (L2-регуляризация). Теперь мы познакомимся с еще одним способом, характерным для распознавания образов и используемым почти повсеместно при обработке изображений с применением моделей глубокого обучения: способом расширения данных (data augmentation).


# 5 .2 .5 . Расширение данных
# Причиной переобучения является недостаточное количество образцов для обу- чения модели, способной обобщать новые данные. Имея бесконечный объем данных, можно было бы получить модель, учитывающую все аспекты распре- деления данных: эффект переобучения никогда не наступил бы. Прием расши- рения данных реализует подход создания дополнительных обучающих данных из имеющихся путем трансформации образцов множеством случайных преоб- разований, дающих правдоподобные изображения. Цель состоит в том, чтобы на этапе обучения модель никогда не увидела одно и то же изображение дважды. Это поможет модели выявить больше особенностей данных и достичь лучшей степени обобщения.
# Сделать это в Keras можно путем настройки ряда случайных преобразований для изображений, читаемых экземпляром ImageDataGenerator. Начнем с простого примера.

# Листинг 5.11. Настройка расширения данных в ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


# Здесь представлена лишь часть возможных вариантов (полный список вы найдете в документации к фреймворку Keras). Давайте быстро пробежимся по этому коду:

# rotation_range — величина в градусах (0–180), диапазон, в котором будет осу- ществляться случайный поворот изображения;

# width_shift и height_shift — диапазоны (в долях ширины и высоты), в преде- лах которых изображения смещаются по горизонтали и вертикали соответ- ственно;

# shear_range — для случайного применения сдвигового (shearing) преобразо- вания;
# zoom_range — для случайного изменения масштаба внутри изображений;

# horizontal_flip — для случайного переворачивания половины изображения по горизонтали — подходит в случае отсутствия предположений о горизонтальной асимметрии (например, в изображениях реального мира);
# fill_mode — стратегия заполнения вновь созданных пикселов, появляющихся после поворота или смещения по горизонтали/вертикали.

# Модуль с утилитами для обработки изображений
from tensorflow.keras.preprocessing  import image
from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

img_path = fnames[3] # Выбор одного изображения для расширения

img = image.load_img(img_path, target_size=(150,150)) # Чтение изображения и изменение его размеров

x = image.img_to_array(img) # Преобразование в массив Numpy с формой (150, 150, 3)

x = x.reshape((1,) + x.shape) # Изменение формы на (1, 150, 150, 3)

# Генерация пакетов случайно преобразованных изобра- жений. Цикл выполняется бесконечно, поэтому его нужно принудительно прер- вать в какой-то момент!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0])) 
    i += 1
    if i % 4 == 0:
        break 
    
plt.show()

# Если обучить новую сеть с использованием этих настроек расширения данных, она никогда не увидит одно и то же изображение дважды. Однако входные данные по- прежнему будут тесно связаны между собой, потому что получены из небольшого количества оригинальных изображений, — у вас не получится сгенерировать но- вую информацию, вы можете только повторить существующую. Поэтому данного решения недостаточно, чтобы избавиться от эффекта переобучения. Продолжая борьбу с ним, добавим в модель слой Dropout непосредственно перед полносвязным классификатором.


# Листинг 5.13. Определение новой сверточной нейронной сети, включающей в себя слой прореживания
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4),metrics=['acc'])

# Обучим сеть, задействовав расширение данных и прореживание.

# Листинг 5.14. Обучение сверточной нейронной сети с использованием генераторов расширения данных

train_datagen = ImageDataGenerator( 
                                   rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2, 
                                   horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255) # Обратите внимание, что проверочные данные не требуется расширять!

train_generator = train_datagen.flow_from_directory(
 train_dir, #Целевой каталог
 target_size=(150, 150),
 batch_size=32,
 class_mode='binary') # Так как используется функция потерь binary_crossentropy, метки должны быть бинарными

validation_generator = test_datagen.flow_from_directory( validation_dir,
target_size=(150, 150), batch_size=32, class_mode='binary')


history = model.fit( train_generator,
steps_per_epoch=100,
epochs=100, validation_data=validation_generator, validation_steps=50)

model.save('cats_and_dogs_small_2.h5')