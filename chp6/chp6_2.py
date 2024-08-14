# 6 .2 . Рекуррентные нейронные сети
# Главной характеристикой всех нейронных сетей, с которыми мы познакомились к данному моменту, таких как полносвязные и сверточные нейронные сети, яв- ляется отсутствие памяти. Каждый вход обрабатывается ими независимо, без сохранения состояния между ними. Чтобы с помощью таких сетей обработать последовательность или временной ряд данных, необходимо передать в сеть всю последовательность целиком, преобразовав ее в единый пакет. Именно так мы по- ступили в предыдущем примере: мы преобразовали все отзывы из IMDB в один большой вектор и обработали его целиком. Такие сети называют сетями прямого распространения (feedforward networks).


# Поэтому текущее состояние первоначально инициализируется вектором с нулевыми значениями элементов, который называют начальным состоянием сети.
# Ниже представлена реализация этой RNN в псевдокоде.

# Листинг 6.19. Реализация RNN в псевдокоде
# 6 .2 . Рекуррентные нейронные сети 229
 
state_t = 0 
for input_t in input_sequence:
    output_t = f(input_t, state_t)
    state_t = output_t
    
# Функцию f можно конкретизировать еще больше: она преобразует входные дан- ные и состояние в выходной результат и параметризуется двумя матрицами, W и U, и вектором смещений. Она напоминает полносвязный слой в сети прямого рас- пространения
Листинг 6.20. Более подробная реализация RNN в псевдокоде
state_t = 0
for input_t in input_sequence:
    output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
    state_t = output_t
    
# Листинг 6.21. Реализация сети RNN на основе Numpy
import numpy as np

# Число временных интервалов во входной последовательности
# Размерность пространства входных признаков
# Размерность пространства выходных признаков

timesteps = 100 
input_features = 32
output_features = 64

# Входные данные: случайный шум для простоты примера
  
inputs = np.random.random ((timesteps, input_features))
#Начальное состояние: вектор
# с нулевыми значениями элементов
state_t = np.zeros((output_features,))

# Создание матриц со случайными весами

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))
#input_t — вектор с формой (входные_признаки,)
successive_outputs = [ ]
for input_t in inputs:
    # Объединение входных данных
    # с текущим состоянием (выходными данными на предыдущем шаге)
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    
    successive_outputs.append(output_t)
    # Обновление текущего состояния
    # сети как подготовка к обработке следующего временного интервала
    state_t = output_t
    
# Окончательный результат — двумер- ный тензор с формой (временные_ин- тервалы, выходные_признаки)
final_output_sequence = np.concatenate(successive_outputs, axis=0)


# Рекуррентные сети характеризуются функцией, реализующей один шаг, такой как следующая,
# использованная в данном примере

output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)


# 6 .2 .1 . Рекуррентный слой в Keras


# Процессу, который мы только что реализовали с применением Numpy, соответству- ет фактический слой в Keras — слой SimpleRNN:

from keras.layers import SimpleRNN

# . Выбор режима управляется аргументом return_sequences конструктора. Рассмотрим пример, в котором используется слой SimpleRNN и воз- вращается результат только для последнего временного интервала:


from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()

Следующий пример возвращает полную последовательность состояний:
    
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()


Иногда полезно наложить друг на друга несколько рекуррентных слоев, чтобы увеличить репрезентативность сети. В таких ситуациях все промежуточные слои должны возвращать полные последовательности результатов


model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.summary()


# Теперь попробуем применить такую же модель для решения задачи классификации отзывов к фильмам из набора данных IMDB. Сначала подготовим данные.

# Листинг 6.22. Подготовка данных IMDB


from keras.datasets import imdb
from keras.preprocessing import sequence



max_features = 10000
maxlen = 500
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(
num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')
print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# Обучим простую рекуррентную сеть, состоящую из слоев Embedding и SimpleRNN. Листинг 6.23. Обучение модели со слоями Embedding и SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(input_train, y_train,
epochs=10, batch_size=128, validation_split=0.2)


# Теперь выведем графики изменения величины потерь и точности модели на этапах обучения и проверки (рис. 6.11 и 6.12).
# Листинг 6.24. Вывод результатов

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

# Как вы помните, первое упрощенное решение этой задачи в главе 3 показало точ- ность 88 % на контрольных данных. К сожалению, эта маленькая рекуррентная сеть не смогла достичь того же уровня (показав на этапе проверки точность лишь 85 %). Отчасти проблема объясняется тем, что анализу подвергаются только 500 первых слов в каждом отзыве, а не вся последовательность, следовательно, сеть RNN получает на входе меньше информации, чем модель, с которой мы ее срав- ниваем. Другая причина в том, что слой SimpleRNN плохо подходит для обработки длинных последовательностей, таких как текст.
# Другие типы рекуррентных слоев позволяют добиться более высоких результатов. Рассмотрим несколько таких более продвинутых слоев.


6 .2 .2 . Слои LSTM и GRU


# Переход от SimpleRNN к LSTM: добавление несущего потока
# А теперь о деталях способа вычисления следующего значения в несущем потоке данных: он основывается на трех разных преобразованиях, все три имеют форму ячейки SimpleRNN:

y = activation(dot(state_t, U) + dot(input_t, W) + b)

# Однако эти преобразования имеют свои весовые матрицы, которые мы обозначим индексами i, f и k. Вот что мы имеем (это может показаться необоснованным, но наберитесь терпения, я все объясню позже).


# Листинг 6.25. Реализация архитектуры LSTM в псевдокоде (1/2)

output_t = activation(dot(state_t, Uo) + dot(input_t, Wo) + dot(C_t, Vo) + bo)

i_t = activation(dot(state_t, Ui) + dot(input_t, Wi) + bi)
f_t = activation(dot(state_t, Uf) + dot(input_t, Wf) + bf)
k_t = activation(dot(state_t, Uk) + dot(input_t, Wk) + bk)

# Получим новое переносимое состояние (далее обозначается как c_t), объединив i_t, f_t и k_t.


# Листинг 6.26. Реализация архитектуры LSTM в псевдокоде (1/2) 

# i_t =  это  "входной" (input) вектор, который управляет тем, насколько новая информация (k_t) добавляется к состоянию ячейки.

# k_t = это новая информация, которую нужно добавить к состоянию ячейки.

# c_t = это предыдущее состояние ячейки. 

# f_t =  это "затвор" (forget gate), который управляет тем, насколько из старого состояния(c_t)  будет сохранено 

c_t+1 = i_t * k_t + c_t * f_t



# Если хотите удариться в философию, подумайте о том, что делает каждая из этих операций. Например, можно сказать, что умножение c_t на f_t — это способ пред- намеренного забывания ненужной информацию в несущем потоке данных. А умножение i_t на k_t представляет собой информацию о настоящем, добавляя новую информацию в несущий поток. Но в конечном счете эти интерпретации не имеют
# большого значения, потому что фактическое действие операций определяется содержимым параметризующих их весов, а веса вычисляются непрерывно и заново в каждом цикле обучения, что делает невозможным приписать какую-то конкретную цель той или иной операции.

# Поэтому набор операций, образующих ячейку RNN, лучше рассматривать как набор ограничений в вашем поиске, а не как дизайн в инженерном смысле


# 6 .2 .3 . Пример использования слоя LSTM из Keras


from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense,LSTM


from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
maxlen = 500
batch_size = 32

(input_train, y_train), (input_test, y_test) = imdb.load_data(
num_words=max_features)

input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)



model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
metrics=['acc'])
history = model.fit(input_train, y_train,
epochs=10, batch_size=128, validation_split=0.2)

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

# На этот раз мы достигли точности 89 % на этапе проверки. Неплохой результат: намного лучше, чем с сетью на основе слоя SimpleRNN, что в значительной степени объясняется меньшей подверженностью LSTM проблеме затухания градиента, — и немного лучше, чем при использовании полносвязного решения в главе 3, даже при том, что в этом примере объем обучающих данных был меньше, чем в главе 3. Здесь мы ограничиваем последовательности 500 словами, тогда как в главе 3 в об- учении участвовали полные последовательности.

# Однако этот результат не является впечатляющим для подхода с таким большим объемом вычислений. Почему решение на основе LSTM не смогло добиться лучшего результата? Одна из причин — мы даже не пытались настроить гиперпараметры, такие как размерность векторных представлений или размерность результата, воз- вращаемого слоем LSTM. Другой причиной может быть отсутствие регуляризации. Однако, если быть честными, главная причина в том, что анализ глобальной про- тяженной структуры отзывов (с чем прекрасно справляется LSTM) плохо помогает в решении задачи определения эмоциональной окраски. Такие простые задачи хорошо решаются путем определения частот слов, которые встречаются в отзывах. Именно на этом было основано первое полносвязное решение. Однако существуют другие, намного более сложные задачи обработки естественного языка, где мощь LSTM проявляется более очевидно: например, в диалоговых системах типа «вопрос/ ответ» и в машинном переводе.


# 6 .2 .4 . Подведение итогов
# Теперь вы знаете:
# что такое рекуррентные нейронные сети (RNN) и как они работают;
# что такое LSTM и почему на длинных последовательностях этот подход дает лучшие результаты, чем простое решение на основе RNN;
# как использовать слои RNN в Keras для обработки последовательных данных.
# Далее мы рассмотрим некоторые дополнительные возможности рекуррентных сетей, которые помогут вам извлечь максимальную выгоду из последовательных моделей глубокого обучения.

# LSTM решает эту проблему с помощью специальных "затворов" и "ячейковых состояний", которые позволяют градиентам сохраняться на больших расстояниях и предотвращают их исчезновение или взрыв

# Архитектура LSTM:

# Ячейка (Cell State): Это основное хранилище для долгосрочной информации. Он проходит через весь путь в сети и может сохранять информацию на протяжении долгого времени.
# Затворы (Gates): Они регулируют, какая информация должна быть добавлена, обновлена или удалена из ячейки. LSTM использует три типа затворов:
# Затвор забвения (Forget Gate): Определяет, какая часть информации из ячейки должна быть забыта.
# Затвор входа (Input Gate): Определяет, какая новая информация будет добавлена в ячейку.
# Затвор выхода (Output Gate): Определяет, какая информация из ячейки будет передана на выход.


