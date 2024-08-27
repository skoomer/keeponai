7 .2 .2 . Введение в TensorBoard: фреймворк визуализации TensorFlow


Но как быть с обработкой результатов? В этом вам поможет TensorBoard.


В этом разделе мы познакомимся с TensorBoard, инструментом визуализации, осно- ванным на использовании браузера, входящего в состав TensorFlow. Обратите внима- ние: его можно использовать для исследования моделей Keras, только когда в качестве низкоуровневого механизма обработки тензоров Keras использует TensorFlow.


Основное назначение TensorBoard — помочь визуально наблюдать за проис- ходящим внутри модели в процессе обучения. Отслеживая больший объем ин- формации, нежели просто окончательные потери модели, вы сможете получить более четкое представление о том, что делает или чего не делает модель, и быстрее добиться прогресса. TensorBoard открывает доступ к некоторым замечательным возможностям через самое обычное окно браузера


визуальный мониторинг метрик в ходе обучения;
визуализация архитектуры модели;
вывод гистограмм активаций и градиентов;
исследование векторных представлений в трехмерном пространстве.


# Листинг 7.7. Модель классификации текста для анализа в TensorBoard


import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features =  2000
max_len = 500 

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)


model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128,input_length=max_len,name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu')) 
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['acc'])

# Перед началом использования TensorBoard необходимо создать каталог, куда будут сохраняться файлы журналов, генерируемые этим инструментом.

# Теперь запустим обучение, передав экземпляр TensorBoard в качестве обратного вы- зова. Этот обратный вызов будет записывать события на диск в указанный каталог.

# Листинг 7.9. Обучение модели с обратным вызовом TensorBoard

# log_dir='my_log_dir', Файлы журналов будут сохраняться в этом каталоге
# histogram_freq=1,  Запись гистограммы активаций в каждой эпохе
# embeddings_freq=1, Запись векторных представлений в каждой эпохе



callbacks = [ keras.callbacks.TensorBoard(log_dir='my_log_dir',histogram_freq=1, embeddings_freq=1,

) ]


history = model.fit(x_train, y_train, epochs=20,
batch_size=128, validation_split=0.2, callbacks=callbacks)



После этого можно запустить сервер TensorBoard из командной строки, указав, что тот должен читать журналы, которые в настоящий момент записывает обратный вызов. Утилита tensorboard должна автоматически установиться вместе с фрейм- ворком TensorFlow:
    
$ tensorboard --logdir=my_log_dir



После этого можно запустить браузер, перейти по адресу http://localhost:6006 и посмотреть, как протекает процесс обучения модели



Примечательно, что Keras поддерживает также другой, более ясный способ пред- ставления моделей в виде графов слоев вместо графов операций TensorFlow: утилиту keras.utils.plot_model. Чтобы воспользоваться ею, нужно установить библиотеки для Python pydot и pydot-ng, а также библиотеку graphviz. Посмотрим, что может эта утилита:
    
    
from keras.utils import plot_model
plot_model(model, to_file='model.png')

Этот вызов создаст изображение в формате PNG, представленное на рис. 7.14.



С помощью этой утилиты также можно отобразить информацию о форме слоев в графе. Следующий пример создает изображение с топологией модели, передавая утилите plot_model параметр show_shapes (рис. 7.15):
from keras.utils import plot_model
plot_model(model, show_shapes=True, to_file='model.png')



7 .2 .3 . Подведение итогов
Обратные вызовы Keras дают простую возможность следить за происходящим внутри модели в ходе ее обучения и автоматически предпринимать какие-либо действия, опираясь на ее состояние.
Если в качестве низкоуровневого механизма обработки тензоров используется TensorFlow, появляется возможность использовать TensorBoard — велико- лепный инструмент визуализации процессов, протекающих в модели, в окне браузера. Для его использования с моделями Keras нужно добавлять в модели обратный вызов TensorBoard.