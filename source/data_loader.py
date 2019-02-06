#!/usr/bin/env python3
# filename: data_loader.py
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
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib
from matplotlib import pylab, mlab
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize, getfigs
import matplotlib.image as mpimg
from sklearn.utils import shuffle


#############
# Load Data #
#############

def load_data():

    path = '../data/1-pre-processed/C'
    colon_normal   = glob.glob(path+'/Normal/*.jpg')
    colon_abnormal = glob.glob(path+'/Abnormal/*.jpg')
    dfn = pd.DataFrame()
    dfa = pd.DataFrame()
    dfn['filepath'] = colon_normal
    dfa['filepath'] = colon_abnormal
    dfn['abnormality'] = 0
    dfa['abnormality'] = 1

    # Split into train and test sets.
    # (train will be further split into train and validation using KFold)
    splitPoint = int(0.8*len(colon_abnormal))
    n_train = dfn[:splitPoint]
    n_test  = dfn[splitPoint:]
    a_train = dfa[:splitPoint]
    a_test  = dfa[splitPoint:]
    df_train = pd.concat([n_train,a_train])
    df_test  = pd.concat([n_test,a_test])
    df_train = shuffle(df_train).reset_index(drop=True)
    df_test  = shuffle(df_test).reset_index(drop=True)

    train_set    = df_train
    train_labels = np.array(train_set['abnormality'].iloc[: ])
    train_files  = df_train['filepath'].tolist()

    test_set    = df_test
    test_labels = np.array(test_set['abnormality'].iloc[: ])
    test_files  = df_test['filepath'].tolist()

    return train_set, train_files, train_labels, test_set, test_files


