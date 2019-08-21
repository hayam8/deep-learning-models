# -*- coding: utf-8 -*-
"""
Convolutional Neural Network (CNN)
Deep learning model which classifies whether an image is of a dog or cat.

Created on Fri Aug  9 15:38:35 2019

@author: hayam
"""

# Build CNN

# Import keras libraries & packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense # Addition of fully connected layers

# Initialize CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Second layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Third layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(activation = 'relu', units = 128))
classifier.add(Dense(activation = 'sigmoid', units = 1))

# Compile CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit CNN to images to prevent overfitting
from keras.preprocessing.image import ImageDataGenerator

# Used to augment images of training set
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255) # Preprocess images

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                              target_size = (64, 64),
                                              batch_size = 32,
                                              class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000 / 32,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000 / 32)

from keras.preprocessing import image
import numpy as np

# Single image prediction
test_img = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis = 0) # Preprocess image

result = classifier.predict(test_img)
training_set.class_indices # show indices for classification of cat / dog
# Convert binary result to String prediction result
if(result[0][0] == 1):
    prediction = "Dog"
else: 
    prediction= "Cat"
    



