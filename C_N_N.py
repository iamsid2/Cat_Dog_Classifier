#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:31:55 2018

@author: iamsid
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# step - 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# step -2 -- Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step - 3 -- Flattening
classifier.add(Flatten())

# Step -4 Full Connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# part - 2 -- Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen= ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set', # path/to/data/
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

classifier.fit_generator(
    training_set,
    samples_per_epoch=8000,
    nb_epoch=25,
    validation_data=test_set,
    nb_val_samples=2000
)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/test_set/cats/cat.4003.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result[0][0])
if result[0][0] == 1:
    prediction = 'dog'
else: 
    prediction = 'cat'
print(prediction)

from keras.models import load_model

# Creates a HDF5 file 'my_model.h5'
classifier.save('my_model.h5')

# Returns a compiled model identical to the previous one
classifier = load_model('my_model.h5')