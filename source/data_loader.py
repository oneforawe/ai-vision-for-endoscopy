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
from sklearn.utils import shuffle


#############
# Load Data #
#############

def load_data(path):

    normalImgFiles   = []
    abnormalImgFiles = []
    for dirPath, subDirs, files in os.walk('./Normal'):
        for file in files:
            if file.endswith(".jpg"):
                normalImgFiles.append( os.path.join(dirPath, file) )
    for dirPath, subDirs, files in os.walk('./Abnormal'):
        for file in files:
            if file.endswith(".jpg"):
                abnormalImgFiles.append( os.path.join(dirPath, file) )
    dfn = pd.DataFrame()
    dfa = pd.DataFrame()
    dfn['filepath'] = normalImgFiles
    dfa['filepath'] = abnormalImgFiles
    dfn['abnormality'] = 0
    dfa['abnormality'] = 1

    # Split into train and test sets.
    # (train will be further split into train and validation using KFold)
    splitPoint = int(0.8*len(abnormalImgFiles))
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


