from hyperport import Hyperport
from hyperport.distributions import choice
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import numpy as np

def data():
    max_features = 2000
    max_len = 500

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    return x_train, y_train, x_test, y_test

def create_model(params):
    model = keras.models.Sequential()
    model.add(layers.Embedding(2000, 128, input_length=500, name='embed'))
    
    model.add(layers.Conv1D(params['conv1_filters'], 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    
    model.add(layers.Conv1D(params['conv2_filters'], 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer=params['optimizer'],
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        epochs=params['epochs'],
                        batch_size=params['batch_size'],
                        validation_split=0.2,
                        verbose=2)
    
    validation_acc = np.amax(history.history['val_acc'])
    return {'loss': -validation_acc, 'status': 'ok'}

if __name__ == '__main__':
    hp = Hyperport(
        model=create_model,
        data=data,
        algo='tpe',
        max_evals=3,
        space={
            'conv1_filters': choice([32, 64, 128]),
            'conv2_filters': choice([32, 64, 128]),
            'optimizer': choice(['rmsprop', 'adam', 'sgd']),
            'epochs': choice([10, 20, 30]),
            'batch_size': choice([32, 64, 128])
        }
    )
    
    best_run, best_model = hp.run()
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    x_train, y_train, x_test, y_test = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
