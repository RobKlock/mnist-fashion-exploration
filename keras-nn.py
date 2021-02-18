#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:10:36 2021

@author: Rob Klock
Exploring neural networks and mlps with tensorflow and keras
I followed along heavilty with Hand-on ML by Geron from O'Reily
"""
import os
# This solution is VERY hacky, only use it at your discretion
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tensorflow import keras 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Our dataset contains 28x28 pixel images of various clothing items (fashion MNIST)
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, Y_train_full), (X_test, Y_test) = fashion_mnist.load_data()

# We have to scale our input features since we're using gradient deschent
# we can also convert the data into floats by dividing by 255.0
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = Y_train_full[:5000] / 255.0, Y_train_full[5000:]

# Number to string mapping of object type
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28,28]))
model.add(keras.layers.Dense(300, activation = "relu"))
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(10, activation = "softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=35,
                    validation_data=(X_valid, y_valid), verbose = 1)

pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()