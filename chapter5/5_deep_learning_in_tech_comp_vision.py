
# 5  Глубокое обучение в технологиях
# компьютерного зрения

# Эта глава охватывает следующие темы:
# 9cуть сверточных нейронных сетей;
# 9расширение обучающего набора данных для ослабления эффекта переобу- чения;
# 9использование предварительно обученной сверточной нейронной сети для извлечения признаков;
# 9дообучение предварительно обученной сверточной нейронной сети; 9визуализация процессов обучения сверточных нейронных сетей и принятия
# ими решений .


# 5 .1 . Введение в сверточные нейронные сети

# В листинге 5.1 показано, как выглядит простая сверточная нейронная сеть. Это стек слоев Conv2D и MaxPooling2D. Как она действует, рассказывается чуть ниже.
# Листинг 5.1. Создание небольшой сверточной нейронной сети

# Важно отметить, что данная сеть принимает на входе тензоры с формой (высота_ изображения, ширина_изображения, каналы) (не включая измерение, определяющее пакеты). В данном случае мы настроили сеть на обработку входов с размерами (28, 28, 1), соответствующими формату изображений в наборе MNIST, передав аргумент input_shape=(28, 28, 1) в первый слой.
# Рассмотрим поближе текущую архитектуру сети:

# Как видите, все слои, Conv2D и MaxPooling2D, выводят трехмерный тензор с формой (высота, ширина, каналы). Измерения ширины и высоты сжимаются с ростом глу- бины сети. Количество каналов управляется первым аргументом, передаваемым в слои Conv2D (32 или 64).

from keras import layers
from keras import models

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

# Следующий шаг — передача последнего выходного тензора (с формой (3, 3, 64)) на вход полносвязной классифицирующей сети, подобной той, с которой мы уже знакомы: стека слоев Dense. Эти классификаторы обрабатывают векторы — одно- мерные массивы, тогда как текущий выход является трехмерным тензором. По- этому мы должны прежде преобразовать трехмерный вывод в одномерный и затем добавить сверху несколько слоев Dense.

# Листинг 5.2. Добавление классификатора поверх сверточной нейронной сети

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

Мы реализуем 10-видовую классификацию, используя конечный слой с 10 выхо- дами и функцией активации softmax. Вот как выглядит сеть теперь:
    
Как видите, выходы (3, 3, 64) преобразуются в векторы с формой (576,) перед передачей двум слоям Dense.
Теперь давайте обучим сеть, передав ей цифры из набора MNIST. Далее мы будем повторно использовать большое количество программного кода из главы 2.
Листинг 5.3. Обучение сверточной нейронной сети на данных из набора MNIST

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)) 
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels) 
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)

test_acc

# 0.9911999702453613


# Полносвязная сеть из главы 2 показала точность 97,8 % на контрольных данных, а простенькая сверточная нейронная сеть показала точность 99,3 %: мы уменьшили процент ошибок на 68 % (относительно). Неплохо!
# Но почему такая простая сверточная нейронная сеть работает настолько хорошо в сравнении с полносвязной моделью? Чтобы ответить на этот вопрос, погрузимся в особенности работы слоев Conv2D и MaxPooling2D.

5 .1 .1 . Операция свертывания
# Основное отличие полносвязного слоя от сверточного заключается в следующем: слои Dense изучают гло- бальные шаблоны в пространстве входных признаков (например, в случае с цифрами из набора MNIST это шаблоны, вовлекающие все пикселы), тогда как свер- точные слои изучают локальные шаблоны (рис. 5.1): в случае с изображениями — шаблоны в небольших двумерных окнах во входных данных. В предыдущем примере все такие окна имели размеры 3 × 3.

Эта ключевая характеристика наделяет сверточные нейронные сети двумя важными свойствами:
    
# Шаблоны, которые они изучают, являются инва- риантными в отношении переноса. После изучения определенного шаблона в правом нижнем углу картинки сверточная нейронная сеть сможет распознавать его повсюду: например, в левом верхнем углу. Полносвязной сети пришлось бы изучить шаблон заново, если он появляется в другом месте. Это увеличивает эффективность сверточных сетей в задачах обработки изображений (потому что видимый мир по своей сути является инвариантным в отношении переноса): таким сетям требуется меньше обучающих образцов для получения представлений, обладающих силой обобщения.


Они могут изучать пространственные иерархии шаблонов (рис. 5.2). Первый сверточный слой будет изучать небольшие локальные шаблоны, такие как края, второй — более крупные шаблоны, состоящие из признаков, возвращаемых первым слоем, и т. д. Это позволяет сверточным нейронным сетям эффективно изучать все более сложные и абстрактные визуальные представления (потому что видимый мир по своей сути является пространственно-иерархическим).


Рис. 5.2. Видимый мир формируется пространственными иерархиями видимых модулей: гиперлокальные края объединяются в локальные объекты, такие как глаза или уши, которые, в свою очередь, объединяются в понятия еще более высокого уровня, такие как «кошка»


При использовании слоев Conv2D дополнение настраивается с помощью аргумен- та padding, который принимает два значения: "valid", означающее отсутствие дополнения (будут использоваться только допустимые местоположения окна), и "same", означающее «дополнить так, чтобы выходная карта признаков имела ту же ширину и высоту, что и входная». По умолчанию аргумент padding получает значение "valid".

Шаг свертки
# Другой фактор, который может влиять на размер выходной карты признаков, — шаг свертки. До сих пор в объяснениях выше предполагалось, что центральная клетка окна свертки последовательно перемещается в смежные клетки входной карты. Однако в общем случае расстояние между двумя соседними окнами является на- страиваемым параметром, который называется шагом свертки и по умолчанию равен 1. Также имеется возможность определять свертки с пробелами (strided convolutions) — свертки с шагом больше 1. На рис. 5.7 можно видеть, как извлека- ются шаблоны 3 × 3 сверткой с шагом 2 из входной карты 5 × 5 (без дополнения).


# Использование шага 2 означает уменьшение ширины и высоты карты признаков за счет уменьшения разрешения в два раза (в дополнение к любым изменениям, вызванным эффектами границ). Свертки с пробелами редко используются на практике, хотя могут пригодиться в моделях некоторых типов, поэтому желательно знать и помнить об этой возможности.
# Для уменьшения разрешения карты признаков вместо шага часто использует- ся операция выбора максимального значения из соседних (max-pooling), которую вы видели в примере первой сверточной нейронной сети.


# 5 .1 .2 . Выбор максимального значения из соседних
# (max-pooling)
# В примере сверточной нейронной сети вы могли заметить, что размер карты при- знаков уменьшается вдвое после каждого слоя MaxPooling2D. Например, перед первым слоем MaxPooling2D карта признаков имела размер 26 × 26, но операция выбора максимального значения из соседних уменьшила ее до размера 13 × 13. В этом заключается предназначение данной операции: агрессивное уменьшение разрешения карты признаков, во многом подобное свертке с пробелами.


# С какой целью вообще производится снижение разрешения карты признаков? По- чему бы просто не убрать слои MaxPooling2D и не использовать карты признаков большего размера? Рассмотрим этот вариант. Сверточная основа модели в этом случае будет выглядеть так:


model_no_max_pool = models.Sequential()
model_no_max_pool.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(28, 28, 1)))
model_no_max_pool.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_no_max_pool.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_no_max_pool.summary()


Что не так в этой конфигурации? Две вещи:
Она не способствует изучению пространственной иерархии признаков. Окна 3 × 3 в третьем слое содержат только информацию, поступающую из окон 7 × 7 в исходных данных. Высокоуровневые шаблоны, изученные с помощью свер- точной нейронной сети, будут слишком малы в сравнении с начальными дан- ными, чего может оказаться недостаточно для обучения классификации цифр (попробуйте распознать цифру, посмотрев на нее через окна 7 × 7 пикселов!). Нам нужно, чтобы признаки, полученные от последнего сверточного слоя, со- держали информацию о совокупности исходных данных.
Заключительная карта признаков имеет 22 × 22 × 64 = 30 976 коэффициентов на образец. Это очень много. Если бы вы решили сделать ее плоской, чтобы наложить сверху слой Dense размером 512, этот слой имел бы 15,8 миллиона параметров. Это слишком много для такой маленькой модели и в результате приведет к интенсивному переобучению.


Проще говоря, уменьшение разрешения используется для уменьшения количества коэффициентов в карте признаков для обработки, а также внедрения иерархий пространственных фильтров путем создания последовательных слоев свертки для просмотра все более крупных окон (с точки зрения долей исходных данных, которые они охватывают).

Обратите внимание на то, что операция выбора максимального значения не единственный способ уменьшения разрешения. Как вы уже знаете, на преды- дущих сверточных слоях можно также использовать шаг свертки. Кроме того, вместо выбора максимального значения вы можете использовать операцию вы- бора среднего значения по соседним элементам (average pooling), когда каждый локальный шаблон преобразуется путем взятия среднего значения для каждого канала в шаблоне вместо максимального. Однако операция выбора максималь- ного значения обычно дает лучшие результаты, чем эти альтернативные решения. Причина в том, что признаки, как правило, кодируют пространственное при- сутствие некоторого шаблона или понятия в разных клетках карты признаков, поэтому максимальное присутствие признаков намного информативнее, чем среднее присутствие. Поэтому более разумная стратегия снижения разрешения состоит в том, чтобы сначала получить плотные карты признаков (путем обычной свертки без пробелов), а затем рассмотреть максимальные значения признаков в небольших шаблонах, а не разреженные окна из входных данных (путем свертки с пробелами) или усредненные шаблоны, которые могут привести к пропуску информации о присутствии.

Однако понятие «большой объем данных» весьма относительно, в первую очередь относительно размера и глубины обучаемой сети. Нельзя обучить сверточную нейронную сеть решению сложной задачи на нескольких десятках образцов, а вот нескольких сотен вполне может хватить, если модель невелика и регуляри- зована, а решаемая задача проста. Так как сверточные нейронные сети изучают локальные признаки, инвариантные в отношении переноса, они обладают вы- сокой эффективностью в решении задач распознавания. Обучение сверточной нейронной сети с нуля на очень небольшом наборе изображений дает вполне неплохие результаты, несмотря на относительную нехватку данных, без необ- ходимости конструировать признаки вручную. В данном разделе мы убедимся в этом на практике.


5 .2 .2 . Загрузка данных
# Набор данных «Dogs vs. Cats», который мы будем использовать, не поставляет- ся в составе Keras. Он был создан в ходе состязаний по распознаванию образов в конце 2013-го, когда сверточные нейронные сети еще не заняли лидирующего положения, и доступен на сайте Kaggle. Этот набор можно получить по адресу: www.kaggle.com/c/dogs-vs-cats/data (для этого вам потребуется создать учетную запись на сайте Kaggle, если у вас ее еще нет, но не волнуйтесь, процесс регистра- ции очень прост).


# Этот набор содержит 25 000 изображений кошек и собак (по 12 500 для каждого класса) общим объемом 543 Мбайт (в сжатом виде). После загрузки и распаковки архива мы создадим новый набор, разделенный на три поднабора: обучающий на- бор с 1000 образцами каждого класса, проверочный набор с 500 образцами каждого класса и контрольный набор с 500 образцами каждого класса.

import os, shutil

original_dataset_dir = '/Users/ivan/Vscodebprojects/AIStudy/kaggle_original_data'
base_dir = '/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small'

base_dir = os.path.expanduser('~/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small')
# os.mkdir(base_dir)


train_dir = os.path.join(base_dir, 'train') #Каталоги для 
# os.mkdir(train_dir) обучающего, 

validation_dir = os.path.join(base_dir, 'validation') # проверочного
# os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test') #и конттрольного поднабора 
# os.mkdir(test_dir)

train_cats_dir = '/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small/train/cats'
train_dogs_dir = '/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small/train/dogs'
validation_cats_dir  = '/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small/validation/cats'
validation_dogs_dir  = '/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small/validation/dogs'
test_cats_dir  = '/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small/test/cats'
test_dogs_dir  = '/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small/test/dogs'

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


print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)] 

for fname in fnames:
    #Копирование первых 1000 изображений
    # с кошками в каталог train_cats_dir
    src = os.path.join(original_dataset_dir, fname) 
    dst = os.path.join(train_cats_dir+'/train', fname) 
    shutil.copyfile(src, dst)
    
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)] 
for fname in fnames:
    # Копирование сле- дующих 500 изо- бражений с кош- ками в каталог validation_cats_dir
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname) 
    shutil.copyfile(src, dst)
    
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)] 
for fname in fnames:
    # Копирование сле- дующих 500 изо- бражений с кош- ками в каталог test_cats_dir
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname) 
    shutil.copyfile(src, dst)
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
#     Копирование первых 1000 изображений
# с собаками в каталог train_dogs_dir
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    # Копирование сле- дующих 500 изо- бражений с соба- ками в каталог validation_dogs_dir
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname) 
    shutil.copyfile(src, dst)
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    # Копирование сле- дующих 500 изо- бражений с со- баками в каталог test_dogs_dir
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
Для проверки подсчитаем, сколько изображений оказалось в каждом поднаборе (обучающем/проверочном/контрольном):
    
    
import os, shutil
 original_dataset_dir = '/kaggle_original_data'
 base_dir = '/cats_and_dogs_small'
 base_dir = os.path.expanduser('~/cats_and_dogs_small')
 os.mkdir(base_dir)

 train_dir = os.path.join(base_dir, 'train')
 validation_dir = os.path.join(base_dir, 'validation')
 test_dir = os.path.join(base_dir, 'test')
 train_cats_dir = os.path.join(train_dir, 'cats')
 train_dogs_dir = os.path.join(train_dir, 'dogs')
 validation_cats_dir = os.path.join(validation_dir, 'cats')
 validation_dogs_dir = os.path.join(validation_dir, 'dogs')
 test_cats_dir = os.path.join(test_dir, 'cats')
 test_dogs_dir = os.path.join(test_dir, 'dogs')
 
 
 
from keras import layers
from keras import models
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
model.add(layers.Dense(512, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))

# Настройка модели для обучения
from keras import optimizers

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4),
metrics=['acc'])


# 5 .2 .4 . Предварительная обработка данных
# Как вы уже знаете, перед передачей в сеть данные должны быть преобразованы в тензоры с вещественными числами. В настоящее время данные хранятся в виде файлов JPEG, поэтому их нужно подготовить для передачи в сеть, выполнив сле- дующие шаги:
# 1. Прочитать файлы с изображениями.
# 2. Декодировать содержимое из формата JPEG в таблицы пикселов RGB.
# 3. Преобразовать их в тензоры с вещественными числами.
# 4. Масштабировать значения пикселов из диапазона [0, 255] в диапазон [0, 1] (как вы уже знаете, нейронным сетям предпочтительнее передавать небольшие значения).


# Использование ImageDataGenerator для чтения изображений из каталогов


# from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)#  Масштабировать значения
test_datagen = ImageDataGenerator(rescale=1./255) 

train_m  = '/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small/train/'


train_generator = train_datagen.flow_from_directory(
    train_m, #  Целевой каталог
    target_size=(150, 150), #Привести все изображения к размеру 150 × 150
    batch_size=20,
    class_mode='binary') #Так как используется функция потерь binary_crossentropy, метки должны быть бинарными

validation_m  = '/Users/ivan/Vscodebprojects/AIStudy/cats_and_dogs_small/validation/'


validation_generator = test_datagen.flow_from_directory(
    validation_m,
     target_size=(150, 150),
         batch_size=20,
    class_mode='binary')




for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


# 5.8 Обучение модели с использованием генератора пакетов

history = model.fit(train_generator,
steps_per_epoch=100,
epochs=30, validation_data=validation_generator, validation_steps=50)


# Формирование графиков изменения потерь и точности в процессе обучения

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


# моделей глубокого обучения: способом расширения данных (data augmentation).
# 5 .2 .5 . Расширение данных

datagen = ImageDataGenerator( 
                             rotation_range=40,
                             width_shift_range=0.2, 
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')


# rotation_range — величина в градусах (0–180), диапазон, в котором будет осу- ществляться случайный поворот изображения;
# width_shift и height_shift — диапазоны (в долях ширины и высоты), в преде- лах которых изображения смещаются по горизонтали и вертикали соответ- ственно;
# shear_range — для случайного применения сдвигового (shearing) преобразо- вания;
# zoom_range — для случайного изменения масштаба внутри изображений;
# horizontal_flip — для случайного переворачивания половины изображения по горизонтали — подходит в случае отсутствия предположений о горизонтальной асимметрии (например, в изображениях реального мира);
# fill_mode — стратегия заполнения вновь созданных пикселов, появляющихся после поворота или смещения по горизонтали/вертикали.
Отображение некоторых обучающих изображений, подвергшихся
 случайным преобразованиям
 
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

 
fnames = [os.path.join(train_cats_dir, fname) for
          fname in os.listdir(train_cats_dir)]

img_path = fnames[3]
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)

x = x.reshape((1,) + x.shape)

i= 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
 plt.show()
 
 
 
#  Определение новой сверточной нейронной сети, включающей в себя слой прореживания
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
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
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4),
metrics=['acc'])


train_datagen = ImageDataGenerator( rescale=1./255,
rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,shear_range=0.2,
zoom_range=0.2, horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_m, 
    # Целевой каталог
    target_size=(150, 150),
    batch_size=32, 
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory( validation_m,
target_size=(150, 150), batch_size=32, class_mode='binary')


history = model.fit(
    train_generator,
    steps_per_epoch=100,
epochs=100, 
validation_data=validation_generator,
validation_steps=50)


# 5 .4 .2 . Визуализация фильтров сверточных
# нейронных сетей

# Листинг 5.32. Определение тензора потерь для визуализации фильтра 

from keras.applications import VGG16
from keras import backend as K

import tensorflow as tf
from tensorflow import keras

model = VGG16(weights='imagenet', include_top=False)
layer_name = 'block3_conv1'
filter_index = 0
layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])


# Чтобы реализовать градиентный спуск, необходимо получить градиент потерь от- носительно входа модели. Для этого можно использовать функцию gradients из модуля backend фреймворка Keras.

# Листинг 5.33. Получение градиента потерь относительно входа модели

# Вызов gradients возвращает список тензоров (в данном случае с разме- ром 1). Поэтому сохраняется только первый элемент (тензор)
grads = K.gradients(loss, model.input)[0]
gradient = keras.backend.gradients(loss, model.input)[0]


from tensorflow.keras import backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


model = VGG16(weights='imagenet',
              include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

grads = K.gradients(loss, model.input)[0]




Листинг 5.34. Трюк с нормализацией градиента 


grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) #Добавить 1e–5 перед операцией деления во избежание случайного деления на ноль


# Листинг 5.35. Получение выходных значений Numpy для заданных входных значений Numpy
iterate = K.function([model.input], [loss, grads])
import numpy as np

loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])
# После этого можно записать цикл, реализующий стохастический градиентный спуск.


# Листинг 5.36. Максимизация потерь стохастическим градиентным спуском Начальное изображение
# с черно-белым шумом

input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.


step = 1.
# Величина каждого изменения градиента

for i in range(40):
    # Вычисление значений потерь и градиента

    loss_value, grads_value = iterate([input_img_data])

    input_img_data += grads_value * step  #step 40 шагов градиентного восхождения
# Корректировка входного изображения в направлении максимизации потерь


# В результате получается тензор вещественных чисел с изображением, имеющий форму (1, 150, 150, 3), значения в котором могут не быть целыми числами в диапазоне [0, 255]. Поэтому нужно в заключение превратить этот тензор в ото- бражаемое изображение. Для этого можно воспользоваться следующей простой функцией.


# Листинг 5.37. Функция преобразования тензора в допустимое изображение

def deprocess_image(x):
# Нормализация: получается тензор со средним значением 0 и стандартным отклонением 0,1
    
    x -= x.mean()
    x /= (x.std() + 1e-5) 
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1) # Ограничивает значения диапазоном [0, 1]
    
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')# Преобразует в массив значений RGB 
    return x
    
    
    # Теперь у нас есть все необходимые элементы. Объединим их в функцию на Python, которая будет принимать имя слоя и индекс фильтра и возвращать тензор с до- пустимым изображением, представляющим собой шаблон, который максимизирует активацию заданного фильтра.
    
    # Листинг 5.38. Функция, генерирующая изображение, которое представляет фильтр
    
def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output # Конструирование функции потерь 
    loss = K.mean(layer_output[:, :, :, filter_index]) # максимизирующей активацию n-го фильтра в заданном слое
    
    grads = K.gradients(loss, model.input)[0] #Вычисление градиента входного изображения с учетом этих потерь
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) #Трюк с нормализа- цией: нормализует градиент
    
    iterate = K.function([model.input], [loss, grads]) # Возврат тензоров потерь и градиента для данного входного изображения
    
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128. # Начальное изображение с черно-белым шумом
    
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data]) # 40 шагов градиентного восхождения
        input_img_data += grads_value * step #

    img = input_img_data[0] 
    return deprocess_image(img)



def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)

import matplotlib.pyplot as plt

Взглянем на получившееся изображение (рис. 5.29):
>>> plt.imshow(generate_pattern('block3_conv1', 0))


# Похоже, что фильтр с индексом 0 в слое block3_conv1 отвечает за узор в горошек. А теперь самое интересное: мы можем визуализировать все фильтры во всех сло- ях. Для простоты рассмотрим только первые 64 фильтра в слое и только первые слои в каждом сверточном блоке (block1_conv1, block2_conv1, block3_conv1, block4_conv1, block5_conv1). Расположим получившиеся изображения шаблонов фильтров 64 × 64 в сетке 8 × 8, добавив небольшие черные границы между шабло-
# нами (рис. 5.30–5.33)


# Листинг 5.39. Создание сетки со всеми шаблонами откликов фильтров в слое
layer_name = 'block1_conv1' 
size = 64
margin = 5

results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

for i in range(8):
    for j in range(8): 
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
         
plt.figure(figsize=(20, 20))
plt.imshow(results)
plt.show()


Эти визуальные представления фильтров могут многое рассказать о том, как слои сверточной нейронной сети видят мир: каждый слой в сети обучает свою кол- лекцию фильтров так, чтобы их входы можно было выразить в виде комбинации фильтров. Это напоминает преобразование Фурье, разлагающее сигнал в пакет ко- синусоидных функций. Фильтры в таких пакетах фильтров сверточной нейронной сети становятся все сложнее с увеличением слоя в модели:
фильтры из первого слоя в модели (block1_conv1) кодируют простые направ- ленные контуры и цвета (или, в некоторых случаях, цветные контуры);
фильтры из block2_conv1 кодируют простые текстуры, состоящие из комбина- ций контуров и цветов;
фильтры в более высоких слоях начинают напоминать текстуры, встречающиеся в естественных изображениях, — перья, глаза, листья и т. д.