# src : https://www.kaggle.com/helibu/cnn-and-resnet-car-and-truck-classification/data

import os, cv2
import random

import numpy as np
import tensorflow as tf
from tensorflow.python import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, AvgPool2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib

import warnings

from src.Utils import *

warnings.filterwarnings('ignore')

data_dir = '../input/datasets/FM_Datasets/'


# src : https://www.kaggle.com/helibu/cnn-and-resnet-car-and-truck-classification/data

# On prend les images
images_raw = []
car_types_raw = []
for car_type in ["citerne","remorqueBenne","remorqueBenneDecharge"]:
    car_dir = data_dir + car_type
    car_files = [car_dir + '/' + filename for filename in os.listdir(car_dir)]
    #print(car_files)
    for filename in car_files:
        if filename.endswith('jpg'):
            try:
                images_raw.append(cv2.resize(cv2.imread(filename), (224,224), interpolation=cv2.INTER_CUBIC))
                car_types_raw.append(car_type)
            except Exception as e:
                print(str(e))



images_raw = np.array(images_raw)
car_types_raw = np.array(car_types_raw)

le = LabelEncoder()
car_types_encoded = le.fit_transform(car_types_raw)
car_types_encoded_onehot = np_utils.to_categorical(car_types_encoded)

# affiche exemples d'image
_, ax = plt.subplots(4,3, figsize=(12,12))
for i in range(3):
    for j in range(3):
        int_random = random.randint(0, len(images_raw)-1)
        ax[i,j].imshow(cv2.cvtColor(images_raw[int_random], cv2.COLOR_BGR2RGB))
        ax[i,j].axis('off')
        ax[i,j].set_title(le.inverse_transform((car_types_encoded[int_random], car_types_encoded[(i*10)+j]))[0], size = 20)
for i in range(3): ax[3,i].axis('off')
ax[3,1].set_title('Exemple d\'image dans le datasets', size = 30)

plt.show()

# mélange
images, car_types, car_types_encoded = shuffle(images_raw, car_types_raw, car_types_encoded)
car_types_encoded.resize((images.shape[0],1))
print(car_types_encoded.shape)

# mise en form des données
car_types_encoded = car_types_encoded.reshape((images.shape[0],1))
car_types_2class = np.zeros((images.shape[0],3))
for i in range(images.shape[0]):
    car_types_2class[i][car_types_encoded[i][0]] = 1
# print(car_types_2class[1:100,:])
x_train, x_val, y_train, y_val = train_test_split(images, car_types_2class, test_size=0.2, random_state=0)
x_train = x_train # / 255
x_val = x_val # / 255


# génération du cnn
input_tensor_shape = (224,224,3)

# ResNet50
# model = build_ResNet50(input_tensor_shape)
#
# model.summary()
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])


model = Sequential()

model.add(Conv2D(32, (3, 3),input_shape=(224, 224, 3),strides = (1,1),  padding = 'same',kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(AvgPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3),strides = (1,1),  padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3),strides = (1,1),  padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3),strides = (1,1),  padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu', kernel_initializer='normal'))
model.add(Dense(units=3, activation='sigmoid', kernel_initializer='normal'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

# set train Generator
datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
datagen.fit(x_train)


# trainning process
nb_epoch = 5
batch_size = 128
#checkpointer = ModelCheckpoint('imagenet', verbose=1, monitor='val_acc',save_best_only=True, save_weights_only=True)
fitted_model2 = model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch = x_train.shape[0],
    epochs=nb_epoch,
    validation_data = (x_val, y_val),
)




plt.plot(fitted_model2.history['accuracy'])
plt.plot(fitted_model2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()





plt.figure()
plt.gcf().clear()
plt.plot(fitted_model2.history['loss'])
plt.plot(fitted_model2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()





# On test avec des images internes
test_image_interne = shuffle(images_raw)
predicted = model.predict_classes(test_image_interne)
_, ax = plt.subplots(4,3, figsize=(12,12))
for i in range(3):
    for j in range(3):
        ax[i,j].imshow(cv2.cvtColor(test_image_interne[i*3 + j], cv2.COLOR_BGR2RGB))
        ax[i,j].axis('off')
        ax[i,j].set_title(le.inverse_transform((predicted[i*3 + j], 0))[0], size = 20)
for i in range(3): ax[3,i].axis('off')
ax[3,1].set_title('Prediction avec des images internes', size = 30)

plt.show()


# On prendre les images externes à test
image_a_test = []
car_dir = data_dir + "a_test"
car_files = [car_dir + '/' + filename for filename in os.listdir(car_dir)]
for filename in car_files:
    if filename.endswith('jpg'):
        try:
            image_a_test.append(cv2.resize(cv2.imread(filename), (224,224), interpolation=cv2.INTER_CUBIC))
        except Exception as e:
            print(str(e))
image_a_test = np.array(image_a_test)
image_a_test = shuffle(image_a_test)

# On test avec des images étrangères
predicted = model.predict_classes(image_a_test)
for plot_id in range(int(len(image_a_test) / 9) + 1):
    _, ax = plt.subplots(4,3, figsize=(12,12))
    for i in range(3):
        for j in range(3):
            ax[i,j].axis('off')
            if len(image_a_test) > plot_id*9 + i*3 + j:
                ax[i,j].imshow(cv2.cvtColor(image_a_test[plot_id * 9 + i * 3 + j], cv2.COLOR_BGR2RGB))
                ax[i,j].set_title(le.inverse_transform((predicted[plot_id * 9 + i * 3 + j], predicted[plot_id * 9 + i * 3 + j]))[0], size = 20)
    for i in range(3): ax[3,i].axis('off')
    ax[3,1].set_title('Prediction avec des images étrangères', size = 30)
    plt.show()

