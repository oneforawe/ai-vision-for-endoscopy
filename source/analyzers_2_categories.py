#!/usr/bin/env python3
# filename: analyzers_2_categories.py
# version:  1

import os
import glob
import pickle
import numpy as np
import pandas as pd
import cv2
import h5py
import tensorflow as tf
import keras
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Input, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn import metrics
import matplotlib
from matplotlib import pylab, mlab
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize, getfigs
import matplotlib.image as mpimg
from sklearn.utils import shuffle


#########################
# Prepare for GPU usage #
#########################

config=tf.ConfigProto(device_count={'GPU':1,'CPU':8})
sess=tf.Session(config=config)
keras.backend.set_session(sess)


################
# Define Model #
################

def mobilenet_v2_a(img_dim):
    # base network to be built around:
    base_model = MobileNetV2(input_shape=None,
                             #input_shape=img_dim,
                             alpha=1.0,
                             depth_multiplier=1,
                             include_top=False,
                             weights='imagenet',
                             input_tensor=None,
                             pooling=None
                             #classes=1000
                            )
    for layer in base_model.layers:
        layer.trainable = False
    #for layer in base_model.layers[:153]:
    #    layer.trainable = False
    #for layer in base_model.layers[153:]:
    #    layer.trainable = True

    xi = Input(shape=img_dim)              # input tensor
    x  = BatchNormalization()(xi)          # next layer
    x  = base_model(x)                     # Each x on the right refers to
    x  = Dropout(0.5)(x)                   #  the previous x on the left.
    x  = Flatten()(x)                      #
    xo = Dense(1, activation='sigmoid')(x) # output tensor
    model = Model(inputs=xi, outputs=xo, name='mobilenet_v2_a')
    return model


################################
# Define Train/Save Procedures #
################################

def train_model(input_model, batch_size, epochs, img_size,
                x, y, test, n_fold, kf, breeder_v, j):

    roc_auc      = metrics.roc_auc_score
    preds_train  = np.zeros(len(x), dtype = np.float)
    preds_test   = np.zeros(len(test), dtype = np.float)
    train_scores = []; valid_scores = []

    model = input_model

    os.makedirs(f'./chkpts/{model.name}/{breeder_v}', exist_ok=True)
    os.makedirs(f'./tb_logs/{model.name}/{breeder_v}',exist_ok=True)

    print(f'\n')
    print(f'Training: Run {j:02d}')
    print(f'----------------\n\n')
    os.makedirs(f'./chkpts/{model.name}/{breeder_v}/Run_{j:02d}', exist_ok=True)  # for /weights_fold_{str(i)}.hdf5
    os.makedirs(f'./tb_logs/{model.name}/{breeder_v}/Run_{j:02d}',exist_ok=True)  # for /tb_fold_{str(i)}/etcetera

    # "Fold" counter
    i = 0
    # histories of each fold
    histories_ = []

    for train_index, valid_index in kf.split(x):
        x_train = x.iloc[train_index]; x_valid = x.iloc[valid_index]
        y_train = y[train_index]; y_valid = y[valid_index]

        i += 1
        print(f'Now beginning training for fold {i}\n')

        def augment(src, choice):
            if choice == 0:
                # Rotate 90
                src = np.rot90(src, 1)
            if choice == 1:
                # flip vertically
                src = np.flipud(src)
            if choice == 2:
                # Rotate 180
                src = np.rot90(src, 2)
            if choice == 3:
                # flip horizontally
                src = np.fliplr(src)
            if choice == 4:
                # Rotate 90 counter-clockwise
                src = np.rot90(src, 3)
            if choice == 5:
                # Rotate 180 and flip horizontally
                src = np.rot90(src, 2)
                src = np.fliplr(src)
            return src

        def train_generator():
            while True:
                for start in range(0, len(x_train), batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + batch_size, len(x_train))
                    train_batch = x_train[start:end]
                    for filepath, tag in train_batch.values:
                        img = cv2.imread(filepath)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, img_size)
                        img = augment(img, np.random.randint(6))
                        x_batch.append(img)
                        y_batch.append(tag)
                    x_batch = np.array(x_batch, np.float32) / 255.
                    y_batch = np.array(y_batch, np.uint8)
                    yield x_batch, y_batch

        def valid_generator():
            while True:
                for start in range(0, len(x_valid), batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + batch_size, len(x_valid))
                    valid_batch = x_valid[start:end]
                    for filepath, tag in valid_batch.values:
                        img = cv2.imread(filepath)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, img_size)
                        #img = augment(img, np.random.randint(6))
                        x_batch.append(img)
                        y_batch.append(tag)
                    x_batch = np.array(x_batch, np.float32) / 255.
                    y_batch = np.array(y_batch, np.uint8)
                    yield x_batch, y_batch

        def test_generator():
            while True:
                for start in range(0, len(test), batch_size):
                    x_batch = []
                    end = min(start + batch_size, len(test))
                    test_batch = test[start:end]
                    for filepath in test_batch:
                        img = cv2.imread(filepath)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, img_size)
                        x_batch.append(img)
                    x_batch = np.array(x_batch, np.float32) / 255.
                    yield x_batch

        train_steps = len(x_train) / batch_size
        valid_steps = len(x_valid) / batch_size
        test_steps = len(test) / batch_size

        os.makedirs(f'./tb_logs/{model.name}/{breeder_v}/Run_{j:02d}/tb_fold_{str(i)}',exist_ok=True)
        # for /tb_fold_{str(i)}/logfiles
        # and /weights_fold_{str(i)}.hdf5

        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=1,
                                       verbose=1, min_lr=1e-7),
                     ModelCheckpoint(filepath =
                                     f'./chkpts/{model.name}/{breeder_v}/Run_{j:02d}/weights_fold_{str(i)}.hdf5',
                                     verbose=1, save_best_only=True, save_weights_only=True, mode='auto'),
                     TensorBoard(log_dir = f'./tb_logs/{model.name}/{breeder_v}/Run_{j:02d}/tb_fold_{str(i)}/' )
                    ]

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
                      metrics = ['accuracy'])

        #
        history = model.fit_generator(train_generator(), train_steps, epochs=epochs, verbose=1,
                                      callbacks=callbacks, validation_data=valid_generator(),
                                      validation_steps=valid_steps)
        histories_.append(history)

        model.load_weights(filepath = f'./chkpts/{model.name}/{breeder_v}/Run_{j:02d}/weights_fold_{str(i)}.hdf5' )

        print('Running validation predictions on fold {}'.format(i))
        preds_valid = model.predict_generator(generator=valid_generator(),
                                              steps=valid_steps, verbose=1)[:, 0]

        print('Running train predictions on fold {}'.format(i))
        preds_train = model.predict_generator(generator=train_generator(),
                                              steps=train_steps, verbose=1)[:, 0]

        valid_score = roc_auc(y_valid, preds_valid)
        train_score = roc_auc(y_train, preds_train)
        print('Val Score:   {} for fold {}'.format(valid_score, i))
        print('Train Score: {} for fold {}'.format(train_score, i))

        valid_scores.append(valid_score)
        train_scores.append(train_score)
        print('Avg Train Score:{0:0.5f}, Val Score:{1:0.5f} after {2:0.5f} folds'.format
              (np.mean(train_scores), np.mean(valid_scores), i))

        print('Running test predictions with fold {}'.format(i))

        preds_test_fold = model.predict_generator(generator=test_generator(),
                                                  steps=test_steps, verbose=1)[:, -1]

        preds_test += preds_test_fold

        print('\n\n')

    print('Finished training!')

    # Save
    model.save(f'./chkpts/{model.name}/{breeder_v}/Run_{j:02d}/ModelWhole_trained_{model.name}_{breeder_v}_Run_{j:02d}.hdf5')
    #model = load_model('my_model.hdf5')
    model_json_string = model.to_json()
    with open(f'./chkpts/{model.name}/{breeder_v}/Run_{j:02d}/ModelArch_{model.name}_.json', "w") as json_file:
        json_file.write(model_json_string)
    #model = model_from_json(json_string)
    model.save_weights(f'./chkpts/{model.name}/{breeder_v}/Run_{j:02d}/ModelWeights_trained_{model.name}_{breeder_v}_Run_{j:02d}.hdf5')
    #model.load_weights('my_model_weights.h5')
    # compare with last fold weights (should be same)

    preds_test /= n_fold

    return histories_, preds_test







