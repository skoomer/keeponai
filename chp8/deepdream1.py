import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf

def extract_features(file_path, max_len=500):
    y, sr = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Ensure mel_spectrogram has at least one dimension to pad
    if mel_spectrogram.shape[1] < max_len:
        mel_spectrogram = np.pad(mel_spectrogram, 
                                 ((0, 0), (0, max_len - mel_spectrogram.shape[1])), 
                                 mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :max_len]
    
    return mel_spectrogram

def save_spectrogram(spectrogram, filename):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=22050, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

# Load a pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(None, None, 3))

# Create a model that outputs activations of specific layers
def build_deep_dream_model(base_model, layers_to_maximize):
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    outputs = [layer_dict[layer_name].output for layer_name in layers_to_maximize]
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return dream_model

# Choose layers to maximize
layers_to_maximize = ['mixed2', 'mixed3', 'mixed4']
dream_model = build_deep_dream_model(base_model, layers_to_maximize)


def compute_loss_and_grads(model, image):
    with tf.GradientTape() as tape:
        tape.watch(image)
        activations = model(image)
        loss = tf.reduce_sum([tf.reduce_mean(act) for act in activations])
    grads = tape.gradient(loss, image)
    return loss, grads
def deep_dream_step(image, model, learning_rate=0.01):
    loss, grads = compute_loss_and_grads(model, image)
    grads /= (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-7)
    image += learning_rate * grads
    return image, loss


def deep_dream_audio(file_path, num_iterations=20, learning_rate=0.01):
    # Extract and prepare the audio spectrogram
    original_spectrogram = extract_features(file_path)
    spectrogram_tensor = tf.convert_to_tensor(original_spectrogram[None, ..., None], dtype=tf.float32)
    # Preprocess for the model (expand dims and normalize)
    spectrogram_tensor = tf.image.grayscale_to_rgb(spectrogram_tensor)
    # Run the Deep Dream algorithm
    for i in range(num_iterations):
        spectrogram_tensor, loss = deep_dream_step(spectrogram_tensor, dream_model, learning_rate)
        print(f"Iteration {i+1}/{num_iterations} - Loss: {loss.numpy()}")
    # Postprocess and save the result
    result_spectrogram = tf.squeeze(spectrogram_tensor).numpy()
    result_spectrogram = np.clip(result_spectrogram, -1.0, 1.0)  # Clip to valid range
    save_spectrogram(result_spectrogram, 'deep_dream_spectrogram.png')
    return result_spectrogram

# Example usage
result = deep_dream_audio('/Users/ivan/Vscodebprojects/AIStudy/matt.wav')


file_path = '/Users/ivan/Vscodebprojects/AIStudy/matt.wav'

mel_spectrogram = extract_features(file_path)
save_spectrogram(mel_spectrogram, 'spectrogram.png')