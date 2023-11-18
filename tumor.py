import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import cv2
import random
import pickle
import time

NAME = "brain-tumor-cnn-64x2-{}".format(int(time.time()))

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME))

DATADIR = "/Users/ajolad/Desktop/Extracurricular/BrainTumorClassificationCNN/Data"
CATEGORIES = ["glioma_tumor", "meningioma_tumor", "normal", "pituitary_tumor"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) # path to tumor dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        

IMG_SIZE = 128

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))


training_data=[]

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to tumor dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
            

create_training_data()

random.shuffle(training_data)

X = []
y = []

for features, labels in training_data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)


pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

train_imgs, test_imgs, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), input_shape=X.shape[1:], activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = "relu"), 
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation= "softmax")
])

model.compile(loss="sparse_categorical_crossentropy", 
              optimizer ="rmsprop", 
              metrics = ['accuracy'])

model.summary()

history = model.fit(train_imgs, train_labels, validation_data = (test_imgs, test_labels),batch_size=32, epochs = 18, callbacks = [tensorboard])

model.evaluate(test_imgs, test_labels)

classifications = model.predict(test_imgs)
print(classifications[0])
print(test_labels[0])
