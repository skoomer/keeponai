from hyperas import optim
from hyperas.distributions import choice, uniform
import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.callbacks import TensorBoard
import numpy as np

# Определите функцию для создания модели
def create_model(embedding_dim=128, conv_filters=32, conv_kernel_size=7, pool_size=5, dense_units=1, dropout_rate=0.0):
    model = Sequential()
    model.add(layers.Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=max_len, name='embed'))
    model.add(layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=pool_size))
    model.add(layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(dense_units))
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model

# Определите функцию для получения данных
def data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    return x_train, y_train, x_test, y_test

# Определите функцию для обучения модели
def hyperas_train():
    x_train, y_train, x_test, y_test = data()

    model = create_model(
        embedding_dim=choice([50, 100, 128]),
        conv_filters=choice([16, 32, 64]),
        conv_kernel_size=choice([3, 5, 7]),
        pool_size=choice([2, 5, 10]),
        dense_units=choice([1, 10, 50]),
        dropout_rate=uniform(0, 0.5)
    )
    
    callbacks = [TensorBoard(log_dir='my_log_dir', histogram_freq=1, embeddings_freq=1)]

    history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2, callbacks=callbacks, verbose=0)
    
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test score: {score}, Test accuracy: {acc}')
    return acc

# Запустите оптимизацию гиперпараметров
if __name__ == '__main__':
    best_run, best_model = optim.minimize(
        model=hyperas_train,
        algo=optim.tpe.suggest,
        max_evals=5,
        trials=optimTrials.Trials()
    )
    print(f'Best model accuracy: {best_model}')
