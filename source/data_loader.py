#!/usr/bin/env python3
# filename: data_loader.py
# version:  1

import os
import glob
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def load_data_2class_abnormality(path):

    normalImgFiles   = []
    abnormalImgFiles = []
    for dirPath, subDirs, files in os.walk(path+'/Normal'):
        for file in files:
            if file.endswith(".jpg"):
                normalImgFiles.append( os.path.join(dirPath, file) )
    for dirPath, subDirs, files in os.walk(path+'/Abnormal'):
        for file in files:
            if file.endswith(".jpg"):
                abnormalImgFiles.append( os.path.join(dirPath, file) )
    dfn = pd.DataFrame()
    dfa = pd.DataFrame()
    dfn['filepath'] = normalImgFiles
    dfa['filepath'] = abnormalImgFiles
    dfn['abnormality'] = 0
    dfa['abnormality'] = 1

    # check
    print(f'len(dfn) = {len(dfn)}')
    print(f'len(dfa) = {len(dfa)}')
    print('')
    print(f'len(normalImgFiles) = {len(normalImgFiles)}')
    print(f'len(abnormalImgFiles) = {len(abnormalImgFiles)}')
    print('')
    print(f'(used)  splitPoint = int(0.8*len(abnormalImgFiles)) = {int(0.8*len(abnormalImgFiles))}')
    print(f'(check) splitPoint = int(0.8*len(dfa)) = {int(0.8*len(dfa))}')

    # Split into train and test sets.
    # (train will be further split into train/learn-validation folds with KFold)
    normal_splitPoint   = int(0.8*len(normalImgFiles))
    abnormal_splitPoint = int(0.8*len(abnormalImgFiles))
    n_train = dfn[:normal_splitPoint]
    print(f'len(n_train) = {len(n_train)}')
    n_test  = dfn[normal_splitPoint:]
    print(f'len(n_test) = {len(n_test)}')
    a_train = dfa[:abnormal_splitPoint]
    print(f'len(a_train) = {len(a_train)}')
    a_test  = dfa[abnormal_splitPoint:]
    print(f'len(a_test) = {len(a_test)}')
    df_train = pd.concat([n_train,a_train])
    print(f'len(df_train) = {len(df_train)}')
    df_test  = pd.concat([n_test,a_test])
    print(f'len(df_test) = {len(df_test)}')
    df_train = shuffle(df_train).reset_index(drop=True)
    print(f'len(df_train) = {len(df_train)}')
    df_test  = shuffle(df_test).reset_index(drop=True)
    print(f'len(df_test) = {len(df_test)}')

    train_set    = df_train
    train_labels = np.array(train_set['abnormality'].iloc[: ])
    train_files  = df_train['filepath'].tolist()

    test_set    = df_test
    test_labels = np.array(test_set['abnormality'].iloc[: ])
    test_files  = df_test['filepath'].tolist()

    # check
    print(f'len(train_set) = {len(train_set)}')
    print(f'len(test_set) = {len(test_set)}')

    return train_set, train_files, train_labels, test_set, test_files


