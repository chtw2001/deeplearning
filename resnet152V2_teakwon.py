from keras.applications import ResNet50V2, ResNet152V2, NASNetLarge
from keras.applications.resnet_v2 import preprocess_input
from keras import models, layers
from keras import Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.optimizers import legacy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dropout
import math
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam


# 배치정규화 추가(line 62), 이미지 다양하게 생성, 학습률 0.0001
# train_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_dir = os.path.join('/Users/chtw2001/Downloads/training_croped_car')
val_dir = os.path.join('/Users/chtw2001/Downloads/validation_croped_car')
 
 
train_generator = train_datagen.flow_from_directory(train_dir, batch_size=16, target_size=(224, 224), color_mode='rgb')
val_generator = val_datagen.flow_from_directory(val_dir, batch_size=16, target_size=(224, 224), color_mode='rgb')


# number of classes
K = 5
 
 
input_tensor = Input(shape=(224, 224, 3), dtype='float32', name='input')

# Load ResNet50V2 model
#base_model = ResNet50V2(weights='imagenet', include_top=False, input_tensor=input_tensor)
base_model = ResNet152V2(weights='imagenet', include_top=False, input_tensor=input_tensor)
#base_model = NASNetLarge(weights='imagenet', include_top=False, input_tensor=input_tensor)
# Freeze the convolutional layers 
# resnet50v2 => 190 layers
for layer in base_model.layers: 
    layer.trainable = False

# Global average pooling layer
x = GlobalAveragePooling2D()(base_model.output)

# 배치정규화 => 그레이디언트 소실 줄임 
x = BatchNormalization()(x)

# Dropout layer
x = Dropout(0.5)(x)

# Dense output layer
output_tensor = Dense(K, activation='softmax')(x)

# Create the model
resnet50v2 = Model(inputs=base_model.input, outputs=output_tensor)

# 고속 optimizer 
adam = Adam(learning_rate=0.0001)

# Compile the model
resnet50v2.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Print model summary
resnet50v2.summary()

# Fit the model
checkpointer = ModelCheckpoint('resnet50v2_best_weights.hdf5', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)

history = resnet50v2.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[early_stopping, checkpointer]
)

# Plot the loss and accuracy curves
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Save the trained model
resnet50v2.save('resnet50v2_trained_model.h5')
