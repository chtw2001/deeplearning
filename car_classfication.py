import tensorflow as tf
import os
import sys
import shutil

from tensorflow.python.client import device_lib

from keras import Sequential
from keras.applications.resnet import ResNet50
from keras.layers import Conv2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
# from keras.utils.mul
import keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd

print(tf.config.list_physical_devices ('GPU'))

ROOT_PATH='./dataset'
TRAIN_PATH=f"{ROOT_PATH}/train"
VAL_PATH=f"{ROOT_PATH}/validation"

class Import_data:
    def __init__(self, train_path, val_path):
        self.train_path = train_path
        self.test_path = val_path
        

    def train(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           featurewise_std_normalization=True,
                                           zoom_range=0.2,
                                           channel_shift_range=0.1,
                                           rotation_range=20,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           horizontal_flip=True
                                           )
        train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(224, 224),
            batch_size=32
        )
        val_generator = train_datagen.flow_from_directory(
            self.test_path,
            target_size=(224, 224),
            batch_size=32
        )

        return train_generator, val_generator

class Load_model:
    def __init__(self, train_path, model_name):
        self.num_class = len(os.listdir(train_path))
        self.model_name = model_name

    def resnet_v1_50(self):
        network = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                           pooling='avg')
        return network
    
    def build_network(self):
        network = self.resnet_v1_50()
        
        model = Sequential()
        model.add(network)
        model.add(Dense(1024, activation='relu'))
        #model.add(Dense(2048, activation='relu'))
        model.add(Dense(self.num_class, activation='softmax'))
        model.summary()

        return model
    
class Fine_tunning:
    def __init__(self, train_path, model_name, epoch, val_path, multi_gpu=0):
        self.data = Import_data(train_path, val_path)
        self.train_data, self.val_data = self.data.train()
        self.load_model = Load_model(train_path, model_name)
        self.multi_gpu = multi_gpu
        self.epoch = epoch
        self.model_name = model_name
        self.train_path = train_path
        self.val_path = val_path

    def training(self):
        callback_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        #callback_acc = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)
        data_name = self.train_path.split('/')
        data_name = data_name[len(data_name)-2]
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, decay=1e-5, momentum=0.999, nesterov=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model = self.load_model.build_network()
        save_folder = './model_saved/' + data_name + '/' + self.model_name + '_' + str(self.epoch) + '/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        check_point = ModelCheckpoint(save_folder + 'model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1,
                                        monitor='val_acc', save_best_only=True, mode='auto', callbacks=[callback_loss])
        if self.multi_gpu == 0:
            # model.compile(loss='categorical_crossentropy',
            #               optimizer=optimizer,
            #               metrics=['acc'])
            model.compile(loss='categorical_crossentropy',
                            optimizer=optimizer,
                            metrics=['acc'])
            history = model.fit_generator(
                self.train_data,
                steps_per_epoch=self.train_data.samples / self.train_data.batch_size,
                epochs=self.epoch,
                validation_data=self.val_data,
                validation_steps=self.val_data.samples / self.val_data.batch_size,
                callbacks=[check_point],
                verbose=1)
        # else:
        #     with tf.device('/cpu:0'):
        #         cpu_model = model
        #     model = multi_gpu_model(cpu_model, gpus=self.multi_gpu)
        #     model.summary()
        #     model.compile(loss='categorical_crossentropy',
        #                     optimizer='adam',
        #                     metrics=['acc'])
        #     history = model.fit_generator(
        #         self.train_data,
        #         steps_per_epoch=self.train_data.samples / self.train_data.batch_size,
        #         epochs=self.epoch,
        #         validation_data=self.val_data,
        #         validation_steps=self.val_data.samples / self.val_data.batch_size,
        #         callbacks=[check_point],
        #         verbose=1)
        return history

    def save_accuracy(self, history):
        data_name = self.train_path.split('/')
        data_name = data_name[len(data_name)-2]
        save_folder = './model_saved/' + data_name + '/' + self.model_name + '_' + str(self.epoch) + '/'
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        epoch_list = list(epochs)

        df = pd.DataFrame({'epoch': epoch_list, 'train_accuracy': acc, 'validation_accuracy': val_acc},
                            columns=['epoch', 'train_accuracy', 'validation_accuracy'])
        df_save_path = save_folder + 'accuracy.csv'
        df.to_csv(df_save_path, index=False, encoding='euc-kr')

        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        save_path = save_folder + 'accuracy.png'
        plt.savefig(save_path)
        plt.cla()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        save_path = save_folder + 'loss.png'
        plt.savefig(save_path)
        plt.cla()

        name_list = os.listdir(save_folder)
        h5_list = []
        for name in name_list:
            if '.h5' in name:
                h5_list.append(name)
        h5_list.sort()
        h5_list = [save_folder + name for name in h5_list]
        for path in h5_list[:len(h5_list) - 1]:
            os.remove(path)
        K.clear_session()

def main(save_folder, train_path, val_path):
    model_name="resnet_v50"
    epoch=30

    fine_tunning = Fine_tunning(train_path=train_path,
                                model_name=model_name,
                                val_path = val_path, 
                                epoch=epoch)
    history = fine_tunning.training()
    fine_tunning.save_accuracy(history)

    if not os.path.exists(f'{save_folder}'):
        os.makedirs(f'{save_folder}')
    fileList = os.listdir(save_folder)
    for file in fileList :
        if os.path.isdir(f'{save_folder}/{file}') : continue
        shutil.move(f'{save_folder}/{file}', f'{save_folder}')


if __name__=="__main__":
    save_folder="./model_saved"
    main(save_folder, TRAIN_PATH, VAL_PATH)

