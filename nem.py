import numpy as np
import librosa
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def extract_features(file_path, max_len=500):
    y, sr = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = pad_sequences(mel_spectrogram.T, maxlen=max_len, padding='post', truncating='post')
    return mel_spectrogram

def load_data():
    # Example implementation; replace with your own file paths and labels
    file_paths = ['/Users/ivan/Vscodebprojects/AIStudy/matt.wav', '/Users/ivan/Vscodebprojects/AIStudy/matt2.wav']  # Replace with your file paths
    labels = [0, 1]  # Replace with your labels
    features = [extract_features(file_path) for file_path in file_paths]
    features = np.array(features)
    labels = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return x_train, y_train, x_test, y_test


from keras import layers
from keras.models import Sequential
import keras
import numpy as np
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import tpe, Trials, STATUS_OK

def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(layers.Conv1D(filters={{choice([32, 64, 128])}}, kernel_size=7, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(layers.MaxPooling1D(pool_size=5))
    model.add(layers.Conv1D(filters={{choice([32, 64, 128])}}, kernel_size=7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense({{choice([64, 128])}}, activation={{choice(['relu', 'sigmoid'])}}))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs={{choice([10, 20, 30])}},
                        batch_size={{choice([32, 64, 128])}},
                        validation_split=0.2,
                        verbose=2)
    validation_acc = np.amax(history.history['val_acc'])
    return {'loss': -validation_acc, 'status': STATUS_OK}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=load_data,
                                          algo=tpe.suggest,
                                          max_evals=3,
                                          trials=Trials())
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    
    x_train, y_train, x_test, y_test = load_data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
