import tensorflow as tf
import os
import sys
import shutil
import cv2
import numpy as np
from tensorflow.python.client import device_lib

from keras import Sequential
from keras.applications.resnet import ResNet50, ResNet152
from keras.applications.resnet_v2 import ResNet50V2, ResNet152V2
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge 
from keras.layers import Conv2D, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
# from keras.utils.mul
import keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd

from car_test import save_confusion

print(tf.config.list_physical_devices ('GPU'))

ROOT_PATH='./training_dataset'
TRAIN_PATH=f"{ROOT_PATH}/train"
VAL_PATH=f"{ROOT_PATH}/validation"

maxtirx_n=224

batch_size=32

class Import_data:
    def __init__(self, train_path, val_path):
        self.train_path = train_path
        self.test_path = val_path
    
    def load_images_from_folder(self, folders):
        images = []
        for folder in folders:
            for filename in os.listdir(folder):
                img = cv2.imread(os.path.join(folder,filename))
                if img is not None:
                    img = cv2.resize(img, (maxtirx_n, maxtirx_n))  # resize image
                    images.append(img)
        return images


    def train(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                        #    featurewise_center=True,
                                        #    featurewise_std_normalization=True,
                                        zoom_range=0.2,
                                        channel_shift_range=0.1,
                                        rotation_range=20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True
                                        )
        train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(maxtirx_n, maxtirx_n),
            batch_size=batch_size
        )
        val_generator = train_datagen.flow_from_directory(
            self.test_path,
            target_size=(maxtirx_n, maxtirx_n),
            batch_size=batch_size
        )

        # numpy_data=self.load_images_from_folder(
        #     [f"{self.train_path}/avante", 
        #      f"{self.train_path}/genesis", 
        #      f"{self.train_path}/k5", 
        #      f"{self.train_path}/morning", 
        #      f"{self.train_path}/sonata", 
        #      ])
        # train_datagen.fit(numpy_data)
        return train_generator, val_generator

    def load_images(self, image_paths):
        images = []
        for path in image_paths:
            img = cv2.imread(path)  # 이미지 파일 읽기
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 색상 채널 변환 (BGR -> RGB)
            img = cv2.resize(img, (224, 224))  # 이미지 크기 조정
            images.append(img)
        return np.array(images)

class Load_model:
    def __init__(self, train_path, model_name):
        self.num_class = len(os.listdir(train_path))
        self.model_name = model_name

    def resnet_v1_152(self):
        network = ResNet152(include_top=False, weights='imagenet', input_tensor=None, input_shape=(maxtirx_n, maxtirx_n, 3),
                           pooling='avg')
        return network
    def resnet_v1_50(self):
        network = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(maxtirx_n, maxtirx_n, 3),
                           pooling='avg')
        return network
    def inception_v4(self):
        network = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(maxtirx_n, maxtirx_n, 3),
                            pooling='avg')
        return network
    def resnet_v2_50(self):
        network = ResNet50V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                             pooling='avg')
        return network
    def resnet_v2_152(self):
        network = ResNet152V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(maxtirx_n, maxtirx_n, 3),
                              pooling='avg')
        return network
    def nasnet_large(self):
        network = NASNetLarge(include_top=False, weights='imagenet',input_shape=(maxtirx_n, maxtirx_n, 3),
                              pooling='avg')
        return network

    def build_network(self):
        if self.model_name == 'resnet_v1_50':
            network = self.resnet_v1_50()
        elif self.model_name == 'resnet_v1_152':
            network = self.resnet_v1_152()
        elif self.model_name == 'resnet_v2_50':
            network = self.resnet_v2_50()
        elif self.model_name == 'resnet_v2_152':
            network = self.resnet_v2_152()
        elif self.model_name == 'inception_v4':
            network = self.inception_v4()
        elif self.model_name == 'nasnet_large':
            network = self.nasnet_large()
            
            
        model = Sequential()
        model.add(network)
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
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

    def training(self, lr):
        callback_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        #callback_acc = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)
        data_name = self.train_path.split('/')
        data_name = data_name[len(data_name)-2]
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, decay=1e-5, momentum=0.999, nesterov=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model = self.load_model.build_network()
        save_folder = './model_saved/' + data_name + '/' + "AugImage_" + self.model_name + '_' + str(self.epoch) + '_' + str(lr) + '/'
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
                callbacks=[check_point, callback_loss],
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

    def save_accuracy(self, history, lr):
        data_name = self.train_path.split('/')
        data_name = data_name[len(data_name)-2]
        save_folder = './model_saved/' + data_name + '/' + "AugImage_" + self.model_name + '_' + str(self.epoch) + '_' + str(lr) + '/'
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
        # for path in h5_list[:len(h5_list) - 1]:
        #     os.remove(path)
        K.clear_session()

        # name_list=os.listdir(save_folder)

        # name_list = os.listdir(save_folder)
        # h5_list = []
        # for name in name_list:
        #     if '.h5' in name:
        #         h5_list.append(name)
        # h5_list.sort()
        # h5_list = [save_folder + name for name in h5_list]      
        # save_confusion(h5_list[0], VAL_PATH)


def main(save_folder, train_path, val_path):
    # model_names=['resnet_v2_50', "resnet_v2_152", 'inception_v4']
    # model_names=["resnet_v1_50", 'resnet_v1_152' ,]
    model_names=['inception_v4', 'resnet_v2_152', 'nasnet_large']
    epoches=[100]
    lrs=[0.0001]
    for model_name in model_names:
        if model_name == 'nasnet_large':
            global batch_size
            batch_size=16
            
        for epoch in epoches: 
            for lr in lrs:
                fine_tunning = Fine_tunning(train_path=train_path,
                                            model_name=model_name,
                                            val_path = val_path, 
                                            epoch=epoch)
                history = fine_tunning.training(lr)
                fine_tunning.save_accuracy(history, lr)

                if not os.path.exists(f'{save_folder}'):
                    os.makedirs(f'{save_folder}')
                fileList = os.listdir(save_folder)
                for file in fileList :
                    if os.path.isdir(f'{save_folder}/{file}') : continue
                    shutil.move(f'{save_folder}/{file}', f'{save_folder}')



if __name__=="__main__":
    save_folder="./model_saved"
    main(save_folder, TRAIN_PATH, VAL_PATH)
    

