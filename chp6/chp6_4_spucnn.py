# 6 .4 . Обработка последовательностей с помощью сверточных нейронных сетей



# 6 .4 .1 . Обработка последовательных данных с помощью одномерной сверточной нейронной сети

# vj

# Например, одномерная сверточная сеть, обрабатывающая последовательность символов и использующая окно свертки с размером 5, способна запоминать сло- ва или фрагменты слов длиной до 5 символов и распознавать эти слова в любом контексте во входной последовательности, — то есть одномерная сверточная сеть, обрабатывающая текст посимвольно, способна изучить морфологию слов.

# 6 .4 .2 . Выбор соседних значений в одномерной последовательности данных

# 6 .4 .3 . Реализация одномерной сверточной сети

# В Keras одномерные сверточные сети создаются с помощью слоя Conv1D, интерфейс которого напоминает интерфейс слоя Conv2D. Он принимает на входе трехмерные тензоры с формой (образцы, время, признаки) и возвращает трехмерные тензоры с той же формой. Окно свертки — это одномерное окно на оси времени: оси с ин- дексом 1 во входном тензоре.


# Листинг 6.45. Подготовка данных IMDB 


from keras.datasets import imdb
from keras.preprocessing import sequence


max_features = 10000
max_len = 500
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len) 
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Листинг 6.46. Обучение и оценка простой одномерной сверточной сети
# на данных IMDB

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop


model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))


model.summary()
model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy',
metrics=['acc'])
history = model.fit(x_train, y_train,
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


# 6 .4 .4 . Объединение сверточных и рекуррентных сетей для обработки длинных последовательностей

# Листинг 6.47. Обучение и оценка простой одномерной сверточной сети на данных из набора Jena
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))


model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)


# Рис. 6.30. Объединение одномерной сверточной и рекуррентной сетей для обработки длинных последовательностей

# В примере повторно используется функция-генератор, которая была определена выше (см. листинг 6.33).
# Листинг 6.48. Подготовка генераторов данных с высоким разрешением для набора данных Jena

step = 3
lookback = 720
delay = 144
# Прежде имело значение 6 (один образец для каждого часа);
# теперь имеет значение 3 (один образец для каждых 30 минут) Не изменились

train_gen = generator(float_data, lookback=lookback,
delay=delay,min_index=0, max_index=200000, shuffle=True, step=step)

val_gen = generator(float_data, lookback=lookback,
delay=delay, min_index=200001, max_index=300000, step=step)


test_gen = generator(float_data, lookback=lookback,
delay=delay, min_index=300001, max_index=None, step=step)

val_steps = (300000 - 200001 - lookback) // 128 
test_steps = (len(float_data) - 300001 - lookback) // 128

#  В примере повторно используется функция-генератор, которая была определена выше (см. листинг 6.33).



# Листинг 6.49. Модель, объединяющая одномерную сверточную основу и уровень GRU
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop


model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))

model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)


# Судя по величине потерь на этапе проверки, эта комбинация не дотягивает до решения с регуляризованным слоем GRU, зато она действует намного быстрее. Она просматривает вдвое больше данных, что в этом случае не кажется особенно по- лезным, но может быть важным в других задачах.


# 6 .4 .5 . Подведение итогов
# Вот какие выводы вы должны сделать из всего, что узнали в этом разделе:
# По аналогии с двумерными сверточными сетями, которые прекрасно справля- ются с задачей выделения визуальных шаблонов в двумерном пространстве, одномерные сверточные сети хорошо подходят для выделения временных ша- блонов. В некоторых задачах, особенно в обработке естественного языка, они могут служить более быстрой альтернативой рекуррентным сетям.
# Обычно одномерные сверточные сети структурируются так же, как их дву- мерные сородичи из мира распознавания образов: они состоят из стопки слоев Conv1D и MaxPooling1D, завершающейся слоем GlobalMaxPooling1D или Flatten.
# Поскольку применение рекуррентных сетей является слишком затратным для обработки очень длинных последовательностей, а применение одномерных свер- точных сетей — менее затратным, может оказаться неплохой идея использовать одномерную сверточную сеть для предварительной обработки последователь- ности перед передачей в рекуррентную сеть. Она сократит последовательность и выделит полезное представление для последующей обработки рекуррентной сетью.


# В этой главе вы познакомились со следующими приемами, применимыми к лю- бым наборам данных, от текста до временных последовательностей:
    
# y как токенизировать текст;

# y что такое векторные представления слов и как их использовать;


# что такое рекуррентные сети и как их использовать;

# y как составлять комбинации рекуррентных слоев и использовать двуна- правленные рекуррентные сети для создания мощных моделей обработки последовательностей;

# y как использовать одномерные сверточные сети для обработки последова- тельностей;

# y как объединять одномерные сверточные и рекуррентные сети для обработки длинных последовательностей


# Рекуррентные сети можно использовать для регрессии («прогнозирования будущего»), классификации, выделения аномалий и маркировки последователь- ностей (например, для выделения имен или дат в предложениях).
# Аналогично одномерные сверточные сети можно использовать для реализации машинного перевода (сверточные модели преобразования последовательностей в последовательности (sequence-to-sequence), такие как SliceNeta1), классифи- кации документов и проверки орфографии.
# Если глобальный порядок следования данных в последовательности имеет значе- ние, обрабатывать такие данные предпочтительнее с применением рекуррентной сети. Это относится, например, к временным последовательностям, в которых недавнее прошлое более информативно, чем отдаленное.
# Если глобальный порядок следования данных не имеет решающего значения, для их обработки с успехом можно использовать одномерные сверточные сети, применение которых менее затратно, а полученный результат оказывается по крайней мере не хуже. Это в особенности относится к текстовым данным, где ключевой шаблон, найденный в начале предложения, будет не менее значимым, чем шаблон, найденный в конце.