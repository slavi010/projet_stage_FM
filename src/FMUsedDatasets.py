# src : https://www.kaggle.com/helibu/cnn-and-resnet-car-and-truck-classification/data

import os, cv2
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

from src.Utils import build_ResNet152

warnings.filterwarnings('ignore')

data_dir = '../input/datasets/FM_Datasets/'



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
_, ax = plt.subplots(3,3, figsize=(12,12))
for i in range(3):
    for j in range(3):
        ax[i,j].imshow(cv2.cvtColor(images_raw[(i*10)+j], cv2.COLOR_BGR2RGB))
        ax[i,j].axis('off')

        ax[i,j].set_title(le.inverse_transform((car_types_encoded[(i*10)+j], car_types_encoded[(i*10)+j]))[0], size = 20)
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

# model2 = build_ResNet152(input_tensor_shape)
#
# model2.summary()
# model2.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

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
nb_epoch = 20
batch_size = 50
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



# On test avec les images originales
predicted = model.predict_classes(images_raw)
for plot in range(int(len(images_raw)/9)+1):
    _, ax = plt.subplots(3,3, figsize=(12,12))
    for i in range(3):
        for j in range(3):
            ax[i,j].axis('off')
            if len(images_raw) > plot*9 + i*3 + j:
                ax[i,j].imshow(cv2.cvtColor(images_raw[plot*9 + i*3 + j], cv2.COLOR_BGR2RGB))
                ax[i,j].set_title(le.inverse_transform((predicted[plot*9 + i*3 + j], predicted[plot*9 + i*3 + j]))[0], size = 20)
    plt.show()