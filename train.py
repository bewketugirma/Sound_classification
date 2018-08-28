from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,Activation
from keras.models import Sequential
from keras.optimizers import SGD, Adam 
from sklearn.utils import shuffle
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from python_speech_features import mfcc

def one_hot(class_name, classes):
    index = classes.index(class_name)
    vector = np.zeros(len(classes))
    vector[index] = 1
    return vector
def get_sample(input_path):
    print("loading features...")
    names = os.listdir(input_path)
    tx = []
    ty = []
    for file in names:
        x, y = load_sample(input_path,file)
        tx.append(np.array(x, dtype=float))
        ty.append(y[0])
    tx = np.array(tx)
    ty = np.array(ty)
    tx = sequence.pad_sequences(tx, maxlen=150, padding='post', truncating='post')
    return tx, ty
    
def load_sample(input_path,name):
   
    with open(input_path + name, 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=",")
        x = []
        y = []
        for row in r:
            x.append(row[:-1])
            y.append(row[-1])
    return np.array(x, dtype=float), np.array(y)
X_train, Y_train = get_sample("dataset/new_features/train/")
X_test, Y_test = get_sample("dataset/new_features/test/")
sounds = os.listdir("dataset/datasets/")
print(sounds)
# emotions = ["neutral", "positive"]
Y = []
Y_train = label_binarize(Y_train, sounds)
Y_test = label_binarize(Y_test, sounds)
# y = np.array(Y, dtype=int)
print(Y_train[0])
X_train, Y_train = shuffle(X_train, Y_train, random_state=42)
X_test, Y_test = shuffle(X_test, Y_test, random_state=42)
# normalize the training set
X_train = (X_train - np.mean(X_train))/np.std(X_train)
X_test = (X_test - np.mean(X_test))/np.std(X_test)

model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True),input_shape=(X_train.shape[1], 13)))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.1))
# model.add(Bidirectional(LSTM(64)))
# model.add(Dropout(0.2))
model.add(Dense(Y_train.shape[1]))
model.add(Dropout(0.1))
model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.compile(Adam(lr = 1e-5), 'categorical_crossentropy', metrics=['accuracy'])
#model.summary()
history=model.fit(X_train, Y_train,
          batch_size=100,
          epochs=50,
          validation_data=[X_test, Y_test])

score = model.evaluate(X_test, Y_test, verbose=0)
print("Model has finished. Accuracy:" + str(score[1]) + " and loss:" + str(score[0]))
print("LSTM Error: %.2f%%" % (100-score[1]*100))
model_json = model.to_json()
with open("dataset/models/rnn/sequence_model_new_data2.json", "w") as json_file:
    json_file.write(model_json)
    model.save_weights("dataset/models/rnn/sequence_weight_new_data2.h5")
    print("Model saved to disk\n")
file = open("dataset/models/rnn/sequence_new2.txt", "w")
los = "loss:"+str(score[0])
acc = "accuracy:" + str(score[1])
file.write(los)
file.write("\n")
file.write(acc)
file.close()
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
