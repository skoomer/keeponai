# 8 Генеративное глубокое обучение


# Эта глава охватывает следующие темы:
#     9генерирование текста с помощью LSTM; 
#     9реализация DeepDream;
#     9нейронная передача стиля: 
#     9вариационные автокодировщики; 
#     9генеративно-состязательные сети .


# В этой главе мы с разных сторон рассмотрим потенциал глубокого обучения для расширения творческих возможностей


# Мы рассмотрим приемы генерирования последовательностей данных

#  алгоритм DeepDream и методы создания изображений с использова- нием вариационных автокодировщиков и генеративно-состязательных сетей.




                    # 8 .1 . Генерирование текста с помощью LSTM
                    
                    
                    
# 8 .1 .2 . Как генерируются последовательности данных?

# В примере, представленном далее в этом разделе, мы возьмем слой LSTM, передадим ему строки длиной N символов, извлеченные из текстового корпуса, и обучим его предсказывать символ N + 1. На выходе модель будет возвращать вектор softmax с вероятностями для всех возможных символов: распределение вероятностей для следующего символа. Такой слой LSTM называется языковой нейронной моделью уровня символов.

# - Начальный текст
# The cat sat on the m

# - Начальный текст
# Языковая модель

# - Распределение вероятностей для следующего символа

# - Стратегия выбора

# - Следующий выбранный символ

# - a 


# 8 .1 .3 . Важность стратегии выбора


# Для генерации текста важную роль играет алгоритм выбора следующего симво- ла. Наивное решение — жадный выбор, когда выбирается наиболее вероятный символ. Но такой подход приводит к получению в результате повторяющихся, предсказуемых строк, которые не выглядят связными предложениями.




# Зачем может понадобиться увеличивать или уменьшать случайную составляю- щую? Рассмотрим крайний случай: чисто случайный выбор, когда следующий


# Для управления величиной случайности в процессе выбора введем параметр, который назовем температурой softmax, характеризующий энтропию распреде- ления вероятностей, используемую для выбора: она будет определять степень не- обычности или предсказуемости выбора следующего символа. С учетом значения temperature и на основе оригинального распределения вероятностей (результата функции softmax модели) будет вычисляться новое распределение путем взвеши- вания вероятностей, как показано ниже.


# Листинг 8.1. Взвешивание распределения вероятностей с учетом значения температуры


import numpy as np


# original_distribution — это одномерный массив Numpy значений вероятностей, сумма которых должна быть равна 1
def reweight_distribution(original_distribution, temperature=0.5):  
    distribution = np.log(original_distribution) / temperature 
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)  # 2

# 2 ---Возвращает новую, взвешенную версию оригинального распределения. Сумма вероятностей в новом распределении может получиться больше 1, поэтому разделим элементы вектора на сумму, чтобы получить новое распределение


# Чем выше температура, тем больше энтропия распределения вероятностей и тем более неожиданными и менее структурированными будут генерируемые данные. Чем меньше температура, тем меньше будет величина случайной составляющей и тем более предсказуемыми будут генерируемые данные (рис. 8.2)



# 8 .1 .4 . Реализация посимвольной генерации текста на основе LSTM


# Воплотим эти идеи на практике в реализации с Keras. Первое, что нам понадобит- ся, — это много текстовых данных, на которых можно было бы обучить языковую модель. Для этого можно использовать любой большой текстовый файл или на- бор текстовых файлов, например статьи из Википедии, роман «Властелин колец» и т. д. В данном примере мы используем тексты из произведений Ницше, немецкого философа конца XIX века (в переводе на английский язык). Таким образом, в ре- зультате обучения у нас получится языковая модель, обладающая специфическими особенностями, характерными для произведений Ницше, а не обобщенная модель английского языка.



# Подготовка данных
# Для начала загрузим корпус и преобразуем текст в нижний регистр.


# Листинг 8.2. Загрузка и парсинг исходного текстового файла 


import keras
import numpy as np
path = keras.utils.get_file('nietzsche.txt',
origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

text = open(path).read().lower()
print('Corpus length:', len(text))


# Затем извлечем частично перекрывающиеся последовательности с длиной maxlen, выполним прямое кодирование и упакуем в трехмерный массив Numpy x с формой (последовательности, максимальная_длина, уникальные_символы). Одновременно подготовим массив y с соответствующими целями: векторы с символами, полу- ченные прямым кодированием, которые следуют за каждой извлеченной после- довательностью.


# Листинг 8.3. Векторизация последовательностей символов

maxlen = 60 # Извлечение последовательностей по 60 символов 
step = 3 # Новые последовательности выбираются через каждые 3 символа 
sentences = [] # Хранение извлеченных последовательностей
next_chars = [] # Хранение целей (символов, следующих за последовательностями)

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
    
    
print('Number of sequences:', len(sentences))


chars = sorted(list(set(text))) # Список уникальных символов в корпусе
print('Unique characters:', len(chars))
char_indices = dict((char, chars.index(char)) for char in chars) # Словарь, отображающий уникальные символы в их индексы в списке «chars»


print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)


for i, sentences in enumerate(sentences):
    for t, char in enumerate(sentences):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1 # Прямое кодирование символов в бинарные массивы
    
    
# Конструирование сети

# Модель с единственным слоем LSTM для предсказания
# следующего символа


from keras import layers
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))



# Так как цели имеют формат прямого кодирования, используем для обучения мо- дели функцию потерь categorical_crossentropy.

# Листинг 8.5. Конфигурация компилируемой модели 

optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)





# Обучение модели и извлечение образцов из нее

# Имея обученную модель и фрагмент начального текста, можно сгенерировать но- вый текст, выполнив следующие пункты:
    
# 1. Извлечь из модели распределение вероятностей следующего символа для име- ющегося на данный момент сгенерированного текста.
# 2. Выполнить взвешивание распределения с заданной температурой.
# 3. Выбрать следующий символ в соответствии с вновь взвешенным распределе- нием вероятностей.
# 4. Добавить новый символ в конец текста.
# Вот код, который мы используем для взвешивания оригинального распределения вероятностей, возвращаемого моделью, и извлечения индекса символа (функция выборки).



# Листинг 8.6. Функция выборки следующего символа с учетом прогнозов модели
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature 
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)




# Наконец, следующий цикл повторяет обучение и генерирует текст. Для начала сгенерируем текст, использовав разные температуры после каждой эпохи. Это по- зволит вам увидеть, как меняется генерируемый тест по мере схождения модели и как температура влияет на стратегию выбора



# Листинг 8.7. Цикл генерации текста

import random
import sys


for epoch in range(1, 60):
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1)
start_index = random.randint(0, len(text) - maxlen - 1) #Выбор случайного начального текста
generated_text = text[start_index: start_index + maxlen] #
print('--- Generating with seed: "' + generated_text + '"') # 

for temperature in [0.2, 0.5, 1.0, 1.2]:
    print('------ temperature:', temperature)
    sys.stdout.write(generated_text)
    
    for i in range(400): # gенерация 400 символов, начиная с начального текста
        sampled = np.zeros((1, maxlen, len(chars))) # Прямое кодирование символов, сгенерирован- ных до сих пор
        for t, char in enumerate(generated_text):#
            sampled[0, t, char_indices[char]] = 1.#
            
        preds = model.predict(sampled, verbose=0)[0] # Выбор следующего символа
        next_index = sample(preds, temperature)#
        next_char = chars[next_index]#
        
        generated_text += next_char
        generated_text = generated_text[1:]
        sys.stdout.write(next_char)

    
#     8 .1 .5 . Подведение итогов
# Обучая модель для предсказания следующего токена по предшествующим, можно генерировать последовательности дискретных данных.
# В случае с текстом такая модель называется языковой моделью; она может быть основана на словах или символах.
# Выбор следующего токена требует баланса между мнением модели и случай- ностью.
# Обеспечить такой баланс можно введением понятия температуры; всегда про- буйте разные температуры, чтобы найти правильную.





import numpy as np



def reweight_distribution(original_distribution, temperature=0.5):  
    distribution = np.log(original_distribution) / temperature 
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)  # 2



import keras
import numpy as np
path = keras.utils.get_file('nietzsche.txt',
origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

text = open(path).read().lower()
print('Corpus length:', len(text))



maxlen = 60 # Извлечение последовательностей по 60 символов
step = 3 # Новые последовательности выбираются через каждые 3 символа
sentences = [] #Хранение извлеченных последовательностей
next_chars = [] # Хранение целей (символов, следующих за последовательностями)

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
    
    
print('Number of sequences:', len(sentences))


chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
char_indices = dict((char, chars.index(char)) for char in chars) #ловарь, отображающий уникальные символы в их индексы в списке «chars»


print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)


for i, sentences in enumerate(sentences):
    for t, char in enumerate(sentences):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1 #Прямое кодирование символов в бинарные массивы
    
# 8.4 Модель с единственным слоем LSTM для предсказания
# следующего символа

from keras import layers
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

# Листинг 8.5. Конфигурация компилируемой модели

optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Обучение модели и извлечение образцов из нее
# Имея обученную модель и фрагмент начального текста, можно сгенерировать но- вый текст, выполнив следующие пункты:
# 1. Извлечь из модели распределение вероятностей следующего символа для име- ющегося на данный момент сгенерированного текста.
# 2. Выполнить взвешивание распределения с заданной температурой.
# 3. Выбрать следующий символ в соответствии с вновь взвешенным распределе- нием вероятностей.
# 4. Добавить новый символ в конец текста.
# Вот код, который мы используем для взвешивания оригинального распределения вероятностей, возвращаемого моделью, и извлечения индекса символа (функция выборки).


# Листинг 8.6. Функция выборки следующего символа с учетом прогнозов модели

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature 
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)



# Листинг 8.7. Цикл генерации текста

import random
import sys


for epoch in range(1, 60):
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1)
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)
        sys.stdout.flush()  # Обновляем вывод сразу
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)
        
        
        
8 .1 .5 . Подведение итогов
Обучая модель для предсказания следующего токена по предшествующим, можно генерировать последовательности дискретных данных.
В случае с текстом такая модель называется языковой моделью; она может быть основана на словах или символах.
Выбор следующего токена требует баланса между мнением модели и случай- ностью.
Обеспечить такой баланс можно введением понятия температуры; всегда про- буйте разные температуры, чтобы найти правильную.