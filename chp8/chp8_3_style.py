 # 8 .3 . Нейронная передача стиля  ####### передача стиля 
    
    #  Определив математически содержимое и стиль, соответствую- щую функцию потерь для минимизации можно обозначить так:
    
# loss = distance(style(reference_image) - style(generated_image)) + distance(content(original_image) - content(generated_image))

# Здесь distance — это функция нормы, такой как L2-норма, content — функция, принимающая изображение и вычисляющая представление его содержимого, а style — функция, принимающая изображение и вычисляющая представление его стиля. Минимизация этой функции потерь приводит к тому, что style(generated_ image) приближается к style(reference_image), а content(generated_image) — к content(original_image), то есть достигается передача стиля, как мы ее опреде- лили.

# 8 .3 .1 . Функция потерь содержимого


# Теперь рассмотрим реализацию оригинального алгоритма нейронной передачи стиля 2015 года с применением Keras. Как вы увидите далее, он имеет много общего с реализацией DeepDream, представленной в предыдущем разделе.

# 8 .3 .3 . Нейронная передача стиля в Keras
# Нейронную передачу стиля можно реализовать с использованием любой обученной сверточной сети. Здесь мы используем сеть VGG19, которую использовали Гатис с коллегами. VGG19 — это упрощенный вариант сети VGG16, представленной в главе 5, с тремя сверточными слоями

# Вот как выглядит весь процесс в общих чертах:
# 1. Настройка сети, которая вычисляет активации слоя VGG19 одновременно для изображения-образца, целевого и сгенерированного изображений.
# 2. Активации, вычисленные по всем трем изображениям, используются для опре- деления общей функции потерь, описанной выше, которая будет минимизиро- ваться для достижения эффекта передачи стиля.
# 3. Настройка процедуры градиентного восхождения для минимизации этой функ- ции потерь


# Листинг 8.14. Определение начальных переменных


from keras.preprocessing.image import load_img, img_to_array

#Путь к изображению, которое будет трансформироваться
target_image_path = 'img/portrait.jpg'

#Путь к изобра- жению с об- разцом стиля
style_reference_image_path = 'img/transfer_style_refer

#Размеры генерируемого изображения
width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)


# Нам понадобится несколько вспомогательных функций для загрузки, а также для предварительной и заключительной обработки изображений перед передачей изо- бражений в сеть VGG19 и после вывода их из сети.
# Листинг 8.15. Вспомогательные функции


import numpy as np
from keras.applications import vgg19


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img
    

# Нулевое центрирование путем удаления среднего значения пиксела из ImageNet. Это отменяет преобразование, выполненное vgg19.preprocess_input

def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68 .
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8') 
    return x
    
    
    # Настроим сеть VGG19. Она принимает на входе пакет из трех изображений: изо- бражение с образцом стиля, целевое изображение и заготовка, куда будет помещено сгенерированное изображение. Заготовка — это символический тензор, значениями которого являются внешние массивы Numpy. Изображение-образец и целевое изо- бражение являются статическими и поэтому определяются как K.constant, тогда как значения в заготовке генерируемого изображения будут изменяться с течением времени
    
# Листинг 8.16. Загрузка предварительно обученной сети VGG19 и применение ее к трем изображениям

from keras import backend as K

target_image = K.constant(preprocess_image(target_image_path))

style_reference_image = K.constant(preprocess_image(style_reference_image_path))

combination_image = K.placeholder((1, img_height, img_width, 3))

input_tensor = K.concatenate([target_image,style_reference_image, combination_image], axis=0)

model = vgg19.VGG19(input_tensor=input_tensor,weights='imagenet',include_top=False)

print('Model loaded.')


# Теперь определим функцию потерь содержимого, которая позволит гарантировать сходство представлений целевого и сгенерированного изображений в верхнем слое сети VGG19.

# Листинг 8.17. Функция потерь содержимого

def content_loss(base, combination):
    return K.sum(K.square(combination - base))
    
    
# Листинг 8.18. Функция потерь стиля

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))
    
# К этим двум компонентам потерь добавляется третий: функция общей потери вариации (total variation loss), которая оперирует пикселами генерируемого изо- бражения. Она стимулирует пространственную целостность генерируемого изо- бражения, что позволяет избежать появления мозаичного эффекта. Ее можно интерпретировать как регуляризацию потерь.

# Листинг 8.19. Функция общей потери вариации

def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, :img_height - 1, 1:, :]) 
    return K.sum(K.pow(a + b, 1.25))
    

# Листинг 8.20. Функция общей потери вариации
# Словарь, отображающий имена слоев в тензоры активаций
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
#Слой, используемый для вычисления потерь содержимого
content_layer = 'block5_conv2' 

#Слой, используемый для вычисления потерь стиля
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1',
'block5_conv1']

# Веса для вычисления среднего взвешенного по компонентам потерь
 
total_variation_weight = 1e-4 #
style_weight = 1. #
content_weight = 0.025 #

# Величина потерь определяется сложением всех компонентов с этой переменной
loss = K.variable(0.)
# Добавление потери содержимого
layer_features = outputs_dict[content_layer]#
target_image_features = layer_features[0, :, :, :] #
combination_features = layer_features[2, :, :, :]#

loss += content_weight * content_loss(target_image_features,
                                      combination_features)
#Add loss style for each targets level                                 
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name] 
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl

#Добавление общей потери вариации
loss += total_variation_weight * total_variation_loss(combination_image)

# Наконец, настроим процесс градиентного восхождения. В оригинальной статье Гатиса оптимизация выполняется с использованием алгоритма L-BFGS, поэтому мы тоже используем его здесь. Это ключевое отличие от примера DeepDream в раз- деле 8.2. Реализация алгоритма L-BFGS уже включена в пакет SciPy, однако она имеет два незначительных ограничения:


# требует передачи значений функции потерь и градиентов в виде двух отдельных функций;

# может применяться только к плоским векторам, тогда как у нас используется
# трехмерный массив с изображением.


Листинг 8.21. Подготовка процедуры градиентного спуска

grads = K.gradients(loss, combination_image)[0]
#Функция для получения значений те- кущих потерь и градиентов
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

class Evaluator(object):
    def __init__(self): 
        self.loss_value = None 
        self.grads_values = None
        
    def loss(self, x):
    assert self.loss_value is None
    x = x.reshape((1, img_height, img_width, 3))
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64') 
    self.loss_value = loss_value
    self.grad_values = grad_values
    return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
        
evaluator = Evaluator()


# Теперь можно запустить процесс градиентного восхождения с использованием реализации алгоритма L-BFGS в SciPy, сохраняя текущее сгенерированное изо- бражение после каждой итерации алгоритма (в данном случае одной итерации соответствуют 20 шагов градиентного восхождения).

# Листинг 8.22. Цикл передачи стиля

from scipy.optimize import fmin_l_bfgs_b 
from scipy.misc import imsave
import time

result_prefix = 'my_result'
iterations = 20
x = preprocess_image(target_image_path)
x = x.flatten()


#  1!  Выполняет оптимизацию L-BFGS по пикселам генерируе- мого изображения, чтобы минимизировать потерю стиля. Обратите внимание: вам нужно передать функцию, кото- рая вычисляет потерю, и функцию, которая вычисляет градиенты, как два отдельных аргумента

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    ### 1! 
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,x,fprime=evaluator.grads,maxfun=20)
    print('Current loss value:', min_val)
    #################saved current gener. img
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    print('Image saved as', fname)
    end_time = time.time() .
    #################
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
    
    

# 8 .3 .4 . Подведение итогов
# Передача стиля заключается в создании нового изображения, которое сохра- няет содержимое целевого изображения и оформлено в стиле изображения- образца.
# Содержимое может сохраняться активациями верхнего слоя сверточной сети.
# Стиль может сохраняться внутренними корреляциями активаций разных слоев.
# Таким образом, передачу стиля в глубоком обучении можно сформулировать как процесс оптимизации, использующий функцию потерь, которая определя- ется предварительно обученной сверточной сетью.
# Начав с этой простой идеи, можно реализовать множество разнообразных ва- риантов.