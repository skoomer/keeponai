# DeepDream — 

# это метод художественной обработки изображений, основанный на использовании представлений, полученных сверточными нейронными сетями. Впервые он был реализован в компании Google летом 2015 года как демонстрация возможностей библиотеки глубокого обучения Caffe (она появилась за несколько
#                                                                                                                                                                                                         месяцев до выхода первой общедоступной версии TensorFlow)1. В интернете он быстро превратился в сенсацию благодаря получаемым с его помощью психоде- лическим картинам (см., например, рис. 8.3), наполненным алгоритмическими иллюзиями, птичьими перьями и собачьими глазами — побочный эффект обучения сверточной сети DeepDream на изображениях из ImageNet, где породы собак и виды птиц представлены шире всего.




# алгоритм DeepDream пытается максимизировать активацию всех слоев, а не только определенного фильтра, тем самым смешивая визуализации большего количества признаков


# вы начинаете не на пустом месте, со случайных входных данных, а с имеющегося изображения, в результате получающиеся эффекты замыкаются на существую- щие визуальные шаблоны, искажая элементы изображения на художественный манер;


# входные изображения обрабатываются в разных масштабах (называемых окта- вами), что улучшает качество визуализации.


# входные изображения обрабатываются в разных масштабах (называемых окта- вами), что улучшает качество визуализации


# 8 .2 .1 . Реализация DeepDream в Keras

# В оригинальной версии DeepDream использо- валась модель Inception, и на практике эта модель известна красиво выглядящими картинками DeepDreams, поэтому мы используем модель Inception V3, входящую в состав Keras.


from keras.applications import inception_v3
from keras import backend as K

# Мы не будем обучать модель, поэтому выполним данную команду, чтобы запретить все операции, имеющие отношение к обучению
K.set_learning_phase(0)

# Конструирование сети Inception V3 без сверточной основы. Модель будет загру- жаться с весами, полученными в результате предварительного обучения на наборе ImageNet

model = inception_v3.InceptionV3(weights='imagenet', include_top=False) 

# Словарь отображает имена слоев в коэффициенты, определяющие вклады слоев в потери, которые мы будем максимизировать. Обратите внимание: имена слоев жестко «зашиты» во встроенное приложение Inception V3. Получить список имен всех слоев можно с помощью model.summary()

layer_contributions = {
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5,
}


# оздается словарь, отображающий имена слоев в их экземпляры

layer_dict = dict([(layer.name, layer) for layer in model.layers])

# Величина потерь определяется  добавлением вклада слоя в эту скалярную переменную
loss = K.variable(0.)

# for layer_name in layer_contributions:
#     coeff = layer_contributions[layer_name]
#     activation = layer_dict[layer_name].output
#     scaling = K.prod(K.cast(K.shape(activation), 'float32'))
#     loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling

for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output #Retrieves the layer’s output
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss = loss + (coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling) 
# Листинг 8.11. Процесс градиентного восхождения

dream = model.input # Этот тензор хранит сгенерированное изображение 
grads = K.gradients(loss, dream)[0] # Вычисление градиентов изображения с учетом потерь
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7) # Нормализация градиентов (важный шаг)

outputs = [loss, grads] # Настройка функции Keras для извлечения значения потерь и градиентов для заданного исходного изображения
fetch_loss_and_grads = K.function([dream], outputs) # 

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value) 
        x += step * grad_values
    return x



import numpy as np

step = 0.01
num_octave = 3
octave_scale = 1.4
iterations = 20

max_loss = 10.

base_image_path =  "/Users/ivan/Vscodebprojects/AIStudy/vv1.jpeg"

img = preprocess_image(base_image_path)

original_shape = img.shape[1:3]
successive_shapes = [original_shape]

for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
    
successive_shapes = successive_shapes[::-1]

original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])


for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img
    
    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape) 
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
    
    
save_img(img, fname='final_dream.png')



# Листинг 8.13. Вспомогательные функции

import scipy
from keras.preprocessing import image

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)
    
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8') 
    return x


8 .2 .2 . Подведение итогов




import scipy
from keras.preprocessing import image

from tensorflow.keras import backend as K

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import scipy
import imageio
import scipy.ndimage
from keras.preprocessing import image

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)

# def save_img(img, fname):
#     pil_img = deprocess_image(np.copy(img))
#     scipy.misc.imsave(fname, pil_img)
    
def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    imageio.imwrite(fname, pil_img)
    
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8') 
    return x


from keras.applications import inception_v3
from keras import backend as K


K.set_learning_phase(0)

model = inception_v3.InceptionV3(weights='imagenet', include_top=False) 

layer_contributions = {
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 2.0,
    'mixed6': 2.0,
    'mixed7': 1.5,
    
}

layer_dict = dict([(layer.name, layer) for layer in model.layers])

loss = K.variable(0.)

for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss = loss + (coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling) 

dream = model.input
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value) 
        x += step * grad_values
    return x



import numpy as np

step = 0.01
num_octave = 3
octave_scale = 1.4
iterations = 20

max_loss = 10.

base_image_path =  "/Users/ivan/Vscodebprojects/AIStudy/vv1.jpeg"

img = preprocess_image(base_image_path)

original_shape = img.shape[1:3]
successive_shapes = [original_shape]

for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i))
                   for dim in original_shape])
    successive_shapes.append(shape)
    
successive_shapes = successive_shapes[::-1]

original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])


for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img
    
    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape) 
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
    
    
    
    
