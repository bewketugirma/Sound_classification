import numpy as np
from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav
import librosa
import os
import csv
from keras.preprocessing import sequence
import pyttsx

def read_audio(filename):
    (sample_rate, signal) = wav.read(filename)
    return signal, sample_rate
def extract_features(audio_path):
    sample, sample_rate = read_audio(audio_path)
    features = mfcc(signal = sample, winlen=0.064, winstep=0.032, nfft=1024)
    X = []
    data = []
    for feat in features:
        X.append(feat)
    X = np.array(X)
    data.append(X)
    return np.array(data)

from keras.models import model_from_json
def load_model(graph_path, weight_path):
    json_file = open(graph_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close
    model = model_from_json(loaded_model_json)
    model.load_weights(weight_path)
    return model
 #give the path where the mode is saved   
model = load_model("dataset/models/rnn/sequence_model_new_data2.json", "dataset/models/rnn/sequence_weight_new_data2.h5")
audio_path="test/sound/1-15689-A-4.wav"
x = np.array(extract_features(audio_path))
print(x.shape)
x = sequence.pad_sequences(x, maxlen=150, padding='post', truncating='post')
# x = x.reshape(x.shape[0], 1, 13)
print(x.shape)
#predict the model with new audio
result = model.predict(x)[0]
print(result)
sounds = os.listdir("dataset/datasets/")
print(sounds)
#change the predicted sound to list
v = result.tolist()
index = v.index(max(v))
print("Predicted sound is ", sounds[index],"confidence = " +str(max(v)) + "%",  "Actuall:" + "Frog")
engine = pyttsx.init()
engine.say('Predicted sound is '+ sounds[index])
engine.runAndWait()