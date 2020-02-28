#test this with the data and a prebuilt library
#https://www.tensorflow.org/tutorials/images/transfer_learning
import numpy as np
import theano
#from keras.datasets import mnist

import os

import random

from random import shuffle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score, f1_score
from keras.models import load_model
from PIL import Image

# example of converting an image with the Keras API

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

# load the image

from numpy import asarray

from os import listdir

from pathlib import Path
from sklearn import preprocessing



DATA_DIR = Path("/home/rsa-key-ml2/train/")  # TODO: Give the path to the directory that contains your images
import os
print(len(os.listdir(DATA_DIR)))
#DATA_DIR=os.getcwd()
#DATA_DIR=PATH("")
import cv2
import glob
import numpy as np
#load images as an nxn numpy array labeled as x, load the labels as an nxn numpy array labeled as y
X_data = []
IMG_SIZE = 50
file_names_pictures=[]
file_names_texts=[]
x, y = [], []
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".png"]:
    image=str(DATA_DIR) +"/"+ str(path)
    #print(image)
    image=cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    t=cv2.resize(image, (40,40))
    x.append(t)
le = LabelEncoder()

le.fit(["red blood cell", "schizont", "ring", "trophozoite"])
for path in [f for f in os.listdir(DATA_DIR) if f[-4:]== ".txt"]:
    text=str(DATA_DIR)+"/"+str(path)
    with open(text) as s:
        label = s.read()
        label= label.splitlines()
        label = le.transform(label)
        y.append(label)

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

x, y = np.array(x), np.array(y)
le = LabelEncoder()
#le.fit(["red blood cell", "difficult", "gametocyte", "trophozoite", "ring", "schizont", "leukocyte"])
#y = le.transform(y)
X_data, Y_data=x, y
#print(X_data.shape())
#print(Y_data.shape())
 # You need to have all images on the same shape before doing this...!!







x_train, y_train, x_test, y_test = train_test_split(X_data, Y_data, test_size=0.70, random_state=4)

print("Hooray?")
#x_train= x_train.reshape(0,1)


np.save("x_train.npy", x_train); np.save("y_train.npy", y_train)

np.save("x_test.npy", x_test); np.save("y_test.npy", y_test)

#train



# %% --------------------------------------- Imports -------------------------------------------------------------------



#estimate 3 hours



# %% --------------------------------------- Set-Up --------------------------------------------------------------------

SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)



# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------

#set it up to loop through all of this; testing range of hyper parameters



LR = .1

N_NEURONS = 10


N_EPOCHS=10

BATCH_SIZE = 1

DROPOUT = 0.2



# %% -------------------------------------- Data Prep ------------------------------------------------------------------

#x, y = np.load("x_train.npy"), np.load("y_train.npy")
#x=x_train
#y=y_train



x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, random_state=SEED, test_size=0.2, stratify=y)

x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
x_train, x_test = x_train/255, x_test/255

#is this the right number of classes?

y_train, y_test = to_categorical(y_train, num_classes=4), to_categorical(y_test, num_classes=4)
y_train, y_test = x_train.reshape(len(y_train), -1), x_test.reshape(len(y_test), -1)

print("successReshaped?")

# %% -------------------------------------- Training Prep ----------------------------------------------------------

model = Sequential()
print("success1")
model.add(Dense(1600, activation="softmax"))
model.add(Dense(1600, activation="relu"))
print("ModelDesigned")
model.compile(optimizer='sgd',loss='mean_squared_error')
print("ModelCompiled")
model.fit(x_train, y_train, batch_size=1, epochs=N_EPOCHS, validation_data=(x_test, y_test),

                    callbacks=[ModelCheckpoint(filepath="/home/rsa-key-ml2/mlp_schmidtModel.pt", monitor="val_loss", save_best_only=True)])
print("ModelFits, you did it")

#print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))


def predict(paths):
    results=[]
    for x in paths:

        image = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
        image= cv2.resize(image, (40, 40))
        X_data = np.array(image)
        x = X_data.reshape(len(x), -1)

        x = x / 255

        X_data = x

        #x = X_data.reshape(len(x), -1)



    # Write any data prep you used during training

    # %% --------------------------------------------- Predict ---------------------------------------------------------

        #change to name of best performing model

        model2 = load_model("mlp_schmidtModel.hdf5")

    # If using more than one model to get y_pred, they need to be named as "mlp_ajafari1.hdf5", ""mlp_ajafari2.hdf5", etc.

        y_pred = np.argmax(model2.predict(X_data), axis=1)

        return y_pred, model2





# %% -------------------------------------------------------------------------------------------------------------------

x_test = ["/home/rsa-key-ml2/train/cell_0.png", "/home/rsa-key-ml2/train/cell_3.png"]  # Dummy image path list placeholder


y_test_pred, *models = predict(x_test)



# %% -------------------------------------------------------------------------------------------------------------------

assert isinstance(y_test_pred, type(np.array([1])))  # Checks if your returned y_test_pred is a NumPy array

assert y_test_pred.shape == (len(x_test),)  # Checks if its shape is this one (one label per image path)

# Checks if the list of unique output label ids is either [0], [0, 1], [0, 1, 2] or [0, 1, 2, 3]

assert list(np.unique(y_test_pred)) in [list(range(i)) for i in range(1, 5)]
