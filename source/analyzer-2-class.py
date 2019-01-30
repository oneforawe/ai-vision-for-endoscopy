#!/usr/bin/env python3
# filename: analyzer-2-class.py

# Source for code modification
# Training and Deploying A Deep Learning Model in Keras MobileNet V2
#  and Heroku: A Step-by-Step Tutorial Part 1
# https://hackernoon.com/tf-serving-keras-mobilenetv2-632b8d92983c

import keras
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Input, Dropout
from keras.models import Model

# dimension of (pre-processed) images
# (when expressed as numpy.ndarray after conversion from jpg)
target_size = 576


def build_model():
    input_tensor = Input(shape=(target_size, target_size, 3))
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(target_size, target_size, 3),
        pooling='avg')

    for layer in base_model.layers:
        layer.trainable = True
        # trainable has to be false in order to freeze the layers

    op = Dense(256, activation='relu')(base_model.output)
    op = Dropout(.25)(op)

    # softmax: good for exclusive classifications
    # activation='softmax': return the highest probability;
    # for example, if class-4 is the highest probability then the result
    # would be something like [0,0,0,0,1,0,0,0,0,0] with 1 in index 4
    output_tensor = Dense(10, activation='softmax')(op)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model

