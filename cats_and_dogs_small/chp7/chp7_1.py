                            # 7    Лучшие практики глубокого обучения
                            #             продвинутого уровня
                            
                            
# Эта глава охватывает следующие темы:
# 9функциональный API фреймворка Keras;
# 9использование обратных вызовов Keras;
# 9использование инструмента визуализации TensorBoard;
# 9важнейшие лучшие практики для разработки современных моделей



# В этой главе рассматривается несколько мощных инструментов, позволяющих конструировать самые современные модели для решения сложных задач. Исполь- зуя функциональный API фреймворка Keras, вы сможете строить графоподобные модели, повторно задействовать один и тот же слой для обработки разных входов и использовать модели Keras подобно функциям на языке Python. Обратные вызо- вы Keras и инструмент визуализации TensorBoard позволяют следить за процессом обучения моделей. Также мы обсудим некоторые другие продвинутые приемы, включая пакетную нормализацию, остаточные связи, оптимизацию гиперпарамет- ров и ансамблирование.


# 7 .1 . За рамками модели Sequential: функциональный API фреймворка Keras

# Все нейронные сети, представленные выше в этой книге, были реализованы с при- менением модели Sequential. Эта модель основана на предположении, что сеть имеет только один вход и только один выход и состоит из линейного стека слоев (рис. 7.1).
# Это общепринятое предположение; данная конфигурация настолько распростране- на, что до настоящего момента мы смогли охватить множество тем и практических применений 


                    # 7 .1 .1 . Введение в функциональный API


# Функциональный API позволяет напрямую манипулировать тензорами и ис- пользовать уровни как функции, которые принимают и возвращают тензоры (чем и обусловлено такое название — функциональный API):


from keras import Input, layers

input_tensor = Input(shape=(32,)) # Тензор
dense = layers.Dense(32, activation='relu') # Слой — это функция
output_tensor = dense(input_tensor) # Вызываемый слой может принимать и возвращать тензор

# Начнем с маленького примера, демонстрирующего простую модель Sequential и ее эквивалент с использованием функционального API:

from keras.models import Sequential, Model 
from keras import layers
from keras import Input

seq_model = Sequential()  #Уже знакомая нам модель Sequential 
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))


# ============================
# Ее функциональный эквивалент

input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)

output_tensor = layers.Dense(10, activation='softmax')(x)



model = Model(input_tensor, output_tensor) # Класс Model превращает входной и выходной тензоры в модель

model.summary()


# Компиляция, обучение и оценка такого экземпляра Model выглядит точно так же, как при использовании модели Sequential

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

import numpy as np 
# Генерация фиктивных данных для обучения

x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))


model.fit(x_train, y_train, epochs=10, batch_size=128)


score = model.evaluate(x_train, y_train) # Оценка модели


# 7 .1 .2 . Модели с несколькими входами


# Следующий пример демонстрирует, как создать модель с помощью функциональ- ного API. В нем создаются две независимые ветви, входной текст и вопрос кодируются в векторные представления; затем эти векторы объединяются и, наконец,
# поверх объединенного представления добавляется классификатор softmax.

# ====================================================================================================
# Листинг 7.1. Реализация модели «вопрос/ответ» с двумя входами
# с использованием функционального API

# Входной текст  text_input — это последова- тельность целых чисел переменной длины. Обратите внимание на то, что при желании можно задать имя последовательности

# embedded_text Преобразование входного текста в после- довательность векторов с размером 64
# encoded_text Преобразование векторов в единый вектор с помощью уровня LSTM
# embedded_question Та же процедура (с другими экземпляра- ми слоев) повторяется для вопроса

# answer  Добавление классификатора softmax сверху


# model - Создание экземпляра модели с двумя входами и одним выходом

from keras.models import Model 
from keras import layers
from keras import Input

text_vocabulary_size = 10000 
question_vocabulary_size = 10000 
answer_vocabulary_size = 500


text_input = Input(shape=(None,), dtype='int32', name='text')

embedded_text = layers.Embedding( text_vocabulary_size, 64)(text_input)

encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None,), dtype='int32',
name='question')


embedded_question = layers.Embedding( question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question],axis=-1)

answer = layers.Dense(answer_vocabulary_size,activation='softmax')(concatenated) 

model = Model([text_input, question_input], answer)

model.compile(optimizer='rmsprop',
loss='categorical_crossentropy', metrics=['acc'])


# Листинг 7.2. Передача данных в модель с несколькими входами

import numpy as np
num_samples = 1000
max_length = 100


# К вопросам применяется прямое коди- рование, а не преобразова- ние в целые числа
# Создание массива Numpy  with  "test" data 

text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))

question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))

answers = np.zeros(shape=(num_samples, answer_vocabulary_size))

indices = np.random.randint(0, answer_vocabulary_size, size=num_samples)

for i, x in enumerate(answers):
    x[indices[i]] = 1
    
model.fit([text, question], answers, epochs=10, batch_size=128) #   передача списка входов 

model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128) #  передача с помощью словоря 


# Оценка модели
loss, accuracy = model.evaluate([text, question], answers, batch_size=128)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# ========================================================================================================================================================================================================

# 7 .1 .3 . Модели с несколькими выходами

# Функциональный API также можно использовать для создания моделей с не- сколькими выходами (или головами). Простейшим примером может служить сеть, пытающаяся одновременно предсказать разные свойства данных, например при- нимающая на входе последовательность постов из социальной сети от некоторой анонимной персоны и пытающаяся предсказать характеристики этой персоны, такие как возраст, пол и уровень доходов (рис. 7.7).


            # Листинг 7.3. Реализация модели с тремя выходами с использованием
                        # функционального API
                        
                        
from keras import layers
from keras import Input
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10
posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)

x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)


age_prediction = layers.Dense(1, name='age')(x) # Обратите внимание: для выход- ных слоев определены имена

income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)

gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input,
[age_prediction, income_prediction, gender_prediction])

# Важно отметить, что для обучения такой модели необходима возможность задавать разные функции потерь для разных выходов: например, определение возраста — это задача скалярной регрессии, но определение пола — задача бинарной классификации, требующая отдельной процедуры обучения. Однако из-за того, что градиентный спуск требует минимизации скаляра, эти функции потерь должны объединяться в единственное значение. Простейший способ объединения потерь — их суммиро- вание. В Keras для этого можно передать функции compile список или словарь с раз- ными объектами для разных выходов; в результате значения потерь будут суммиро- ваться в общее значение потери, которое будет минимизироваться в ходе обучения.


# Листинг 7.4. Параметры компиляции модели с несколькими выходами:
# несколько функций потерь


model.compile(optimizer='rmsprop',
loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])

# Эквивалентное решение (возможно, только если определены имена выходных слоев)
model.compile(optimizer='rmsprop', loss={'age': 'mse',
'income': 'categorical_crossentropy', 'gender': 'binary_crossentropy'})


# Обратите внимание: несбалансированные вклады потерь приведут к созданию пред- ставления, оптимизированного преимущественно для задачи с наибольшей потерей, в ущерб другим задачам. Чтобы исправить эту проблему, можно присвоить разные степени важности значениям потерь, вносящим вклад в общую потерю. Это может пригодиться, когда значения потерь имеют разные масштабы. Например, средняя квадратичная ошибка (Mean Squared Error, MSE), используемая как функция по- терь в задаче определения возраста, обычно принимает значение около 3–5, тогда как перекрестная энтропия, используемая в задаче определения пола, может коле- баться около величины 0,1. Чтобы в такой ситуации сбалансировать вклад разных потерь, можно присвоить вес 10 перекрестной энтропии и вес 0,25 

# Листинг 7.5. Параметры компиляции модели с несколькими выходами: взвешивание потерь


model.compile(optimizer='rmsprop',loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'], loss_weights=[0.25, 1., 10.])

# Эквивалентное решение (возможно, только если определены имена выходных слоев)

model.compile(optimizer='rmsprop', loss={'age': 'mse',
'income': 'categorical_crossentropy',
'gender': 'binary_crossentropy'}, loss_weights={'age': 0.25,
'income': 1., 'gender': 10.})

Так же как в случае с моделями, имеющими несколько входов, передавать обуча- ющие данные в модель можно либо в виде списка массивов Numpy, либо в виде словаря с их именами


# Листинг 7.6. Передача данных в модель с несколькими выходами

# Предполагается, что age_targets, income_ targets и gender_targets — это массивы Numpy

model.fit(posts, [age_targets, income_targets, gender_targets],
epochs=10, batch_size=64)

model.fit(posts, {'age': age_targets, 'income': income_targets,
'gender': gender_targets}, epochs=10, batch_size=64)


# 7 .1 .4 . Ориентированные ациклические графы уровней

