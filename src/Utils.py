import os

from keras import Model
from keras.applications import ResNet152
import requests
from keras.layers import AvgPool2D, Dropout, Dense, Flatten


def build_ResNet152(input_tensor_shape):
    base_model = ResNet152(weights='imagenet', include_top=False, input_shape=input_tensor_shape)

    x_model = base_model.output

    x_model = AvgPool2D(name='globalaveragepooling2d')(x_model)

    x_model = Dense(1024, activation='relu', name='fc1_Dense')(x_model)
    x_model = Dropout(0.5, name='dropout_1')(x_model)
    x_model = Flatten()(x_model)
    x_model = Dense(256, activation='relu', name='fc2_Dense')(x_model)
    x_model = Dropout(0.5, name='dropout_2')(x_model)

    predictions = Dense(3, activation='sigmoid', name='output_layer')(x_model)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model




def dl_dmiFM_datasets():
    types = [('citerne', range(1, 10)), ('remorqueBenne', range(1, 19)), ('remorqueBenneDecharge', range(1, 9))]

    for t in types:
        dl_img(
            "http://dmiftp.uqtr.ca/FMeunier/iar6002/ProjetBesnard/VehicleMotionDetection/VehicleMotionDetection/" + t[0] + "%d.jpg",
            "../input/datasets/FM_Datasets/" + t[0] + "/%d.jpg", t[1]
        )


# src : https://stackoverflow.com/questions/30229231/python-save-image-from-url/30229298
def dl_img(url: str, relative_path_dataset: str, range_num_img):
    os.makedirs(os.path.dirname(relative_path_dataset), exist_ok=True)
    for num_image in range_num_img:
        with open(relative_path_dataset % num_image, 'wb') as handle:
            response = requests.get(url % num_image, stream=True)

            if not response.ok:
                print(response)

            for block in response.iter_content(1024):
                if not block:
                    break

                handle.write(block)
