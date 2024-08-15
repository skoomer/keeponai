# 6 Глубокое обучение для текста
# и последовательностей


# Эта глава охватывает следующие темы:
# 9предварительную обработку текстовых данных;
# 9рекуррентные нейронные сети;
# 9одномерные сверточные нейронные сети для обработки последовательностей

# 6 .1 .1 . Прямое кодирование слов и символов

# Листинг 6.1. Прямое кодирование на уровне слов (упрощенный пример)

import  numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index = { } 
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word]= len(token_index) + 1

max_length = 10

results = np.zeros(shape=(len(samples), max_length,max(token_index.values()) + 1))


for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.
        
# Прямое кодирование на уровне символов (упрощенный пример)

import string

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable# Все отображаемые символы ASCII 

token_index = dict(zip(characters, range(1, len(characters) + 1)))

max_length = 50

# results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))

results = np.zeros((len(samples), max_length, len(token_index) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.
        
        # ============= ============= ============= ============= ============= ============= ============= =============
# Листинг 6.3. Использование Keras для прямого кодирования слов

# Создание токенизатора и его настройка на учет
# только 1000 наиболее часто используемых слов

from keras.preprocessing.text import Tokenizer
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples) #<-------Создание индекса всех слов

sequences = tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))


# Прямое кодирование на уровне слов с использованием хеширования (упрощенный пример)


samples = ['The cat sat on the mat.', 'The dog ate my homework.']
dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.


6 .1 .2 . Векторное представление слов


# Конструирование векторных представлений слов с помощью слоя Embedding

# Листинг 6.5. Создание слоя Embedding
# Иначе говоря, векторное представление слов позволяет уместить больший объем информации в меньшее число измерений



# Слой Embedding принимает как минимум два аргумента: количество возможных токенов (в данном случае 1000: 1 + максимальный индекс слова) и размерность пространства (в данном случае 64

from keras.datasets import imdb
from tensorflow.keras import preprocessing
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Flatten, Dense


embedding_layer = Embedding(1000, 64)


max_features = 10000

maxlen = 20 

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = preprocessing.sequence.pad_sequences (x_train, maxlen=maxlen) # 1
x_test = preprocessing.sequence.pad_sequences (x_test, maxlen=maxlen)

# 1  Преобразование списков целых чисел в двумерный тензор с целыми числами и с формой (образцы, максимальная_длина)

Листинг 6.7. Использование слоя Embedding и классификатора данных из IMDB

# 2 Преобразование трехмерного тензора векторов в двумер- ный тензор с формой (образцы, максималь- ная_длина * 8

# 3 Добавление классификатора сверху



model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=8, input_length=maxlen))  # Embedding layer



model.add(Flatten()) # 2


model.add(Dense(1, activation='sigmoid')) # 3
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,epochs=10,batch_size=32,validation_split=0.2)

#  (например, эта мо- дель наверняка расценит оба отзыва — «this movie is a bomb» и «this movie is the bomb» — как отрицательные1).

Использование предварительно обученных векторных представлений слов

Word2vec  and GloVe  это готовые представления 


6 .1 .3 . Объединение всего вместе: от исходного текста к векторному представлению слов


# Загрузка данных из IMDB в виде простого текста
# Сначала загрузите архив с исходным набором данных IMDB, доступный по адресу http://mng.bz/0tIo. Распакуйте его.
# Теперь соберем отдельные обучающие отзывы в список строк, по одной строке на отзыв. Также соберем метки отзывов (положительный/отрицательный) в список labels



# Листинг 6.8. Обработка меток из исходного набора данных IMDB 

import os
imdb_dir = '/Users/ivan/Vscodebprojects/AIStudy/chp6/aclImdb' 

train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
                
# Токенизация данных

#  добавим следующий трюк: ограничим набор обучающих данных первыми 200 образцами. Другими словами, мы попытаемся получить модель классификации отзывов, обу- чив ее всего на 200 примерах.

# Листинг 6.9. Токенизация текста из исходного набора данных IMDB

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
import numpy as np

maxlen = 100 # Отсечение остатка отзывов после 100-го слова
training_samples = 200 # Обучение на выборке из 200 образцов 
validation_samples = 10000 #Проверка на выборке из 10 000 образцов
max_words = 10000  #Рассмотрение только 10 000 наиболее часто используемых слов

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# Разбивка данных на обучающую и провероч- ную выборки, но перед этим данные пере- мешиваются, поскольку отзывы в исходном наборе упорядочены (сначала следуют отри- цательные, а потом положительные)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]


x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

# Загрузка векторного представления GloVe
# Откройте в браузере страницу https://nlp.stanford.edu/projects/glove и загрузите векторные представления, предварительно обученные на данных из англоязычной Википедии за 2014 год. Этот ZIP-архив размером 822 Мбайт с именем glove.6B.zip содержит 100-мерные векторы с 400 000 слов (токенов). Распакуйте его.

# Предварительная обработка векторных представлений
# Обработаем распакованный файл .txt и создадим индекс, отображающий слова (в виде строк) в их векторные представления (в виде векторов с числами).



# Листинг 6.10. Обработка файла с векторными представлениями слов GloVe

glove_dir = '/Users/ivan/Vscodebprojects/AIStudy/chp6/glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt')) 

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs


f.close()

print('Found %s word vectors.' % len(embeddings_index))


# Теперь создадим матрицу векторных представлений, которую можно будет пере- дать на вход слоя Embedding. Это должна быть матрица с формой (максималь- ное_число_слов, размерность_представления), каждый элемент i которой содержит вектор с размером, равным размерности представления, соответствующий слову с индексом i в индексе (созданном в ходе токенизации). Обратите внимание: ин- декс 0 не должен соответствовать никакому слову или токену — это пустой элемент.


# Листинг 6.11. Подготовка матрицы векторных представлений слов GloVe 
# Словам, отсутствующим
# в индексе представлений, будут соответствовать векторы с нулевыми зна- чениями

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word) 
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
            

# Определение модели
# Используем модель с той же архитектурой, как было показано выше.


# Листинг 6.12. Определение модели

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen)) 
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# Загрузка представлений GloVe в модель

# Уровень Embedding имеет единственную весовую матрицу: двумерную матрицу с вещественными числами, каждый i-й элемент которой — это вектор, связанный с i-м словом в индексе. Все довольно просто. Загрузим подготовленную матрицу GloVe в слой Embedding — первый слой модели.
# Листинг 6.13. Загрузка предварительно обученных векторных представлений слов в слой Embedding


model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# Мы также заморозили слой Embedding (присвоив атрибуту trainable значение False). Причины те же, что были описаны, когда мы знакомились с особенностя- ми применения предварительно обученных сверточных нейронных сетей: переобучение


# Обучение и оценка модели
# Скомпилируем и обучим модель.
# Листинг 6.14. Обучение и оценка

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
metrics=['acc'])
history = model.fit(x_train, y_train,
epochs=10,
batch_size=32, validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')



# Листинг 6.15. Вывод результатов 

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

# Модель быстро достигает состояния переобучения, что неудивительно при таком малом объеме обучающих данных. По этой причине оценка точности демонстри- рует высокую изменчивость, но все же достигает уровня 50 %.
# Имейте в виду, что ваши результаты могут несколько отличаться от представ- ленных, поскольку при таком небольшом объеме обучающих данных результаты сильно зависят от того, какие именно 200 образцов попадут в обучающую выборку при случайном выборе. Если вы получили худший результат, чем мы, попробуйте ради эксперимента отобрать другой набор из 200 случайных образцов (в реальной жизни у вас не будет такой возможности).
********** ---- ======= ******************** ---- ======= **********
# Листинг 6.16. Обучение той же модели без использования уже обученных векторных представлений
********** ---- ======= ******************** ---- ======= **********

from keras.models import Sequential 
from keras.layers import Embedding , Flatten , Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
metrics=['acc']) 

history = model.fit(x_train, y_train,
epochs=10,
batch_size=32, validation_data=(x_val, y_val))



Листинг 6.17. Токенизация данных из контрольной выборки 

test_dir = os.path.join(imdb_dir, 'test')
labels = [] 
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type) 
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0) 
            else:
                labels.append(1)
                
                
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

А затем загрузим и оценим первую модель.


Листинг 6.18. Оценка модели на контрольном наборе данных 

model.load_weights('/Users/ivan/Vscodebprojects/AIStudy/pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)



# 6 .1 .4 . Подведение итогов
# Теперь вы умеете:
# преобразовывать текст в нечто, что можно передать в нейронную сеть;
# использовать слой Embedding в модели Keras для обучения специализированных векторных представлений токенов;
# использовать предварительно обученные векторные представления слов для получения дополнительных выгод в небольших задачах обработки естествен- ного языка