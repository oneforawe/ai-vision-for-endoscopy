#!/usr/bin/env python3
# filename: analyzer_breeder.py

import os
import pickle
import cv2
import data_loader as dl
import analyzers_2_categories as a2c
from sklearn.model_selection import KFold
import datetime


def main():
    start_time = datetime.datetime.now()

    ################
    # File Version #
    ################
    breeder_version = "breeder_01"


    #############
    # Load Data #
    #############
    # A: Full data set in original file structure.
    #path = '../data/1-pre-processed/A'
    #data_folder = "data_A"
    # B: 26 images
    #path = '../data/1-pre-processed/B'
    #data_folder = "data_B"
    # C: 200 images (Abnormal=Blood)
    path = '../data/1-pre-processed/C'
    data_folder = "data_C"
    # D: 2000 images
    #path = '../data/1-pre-processed/D'
    #data_folder = "data_D"
    # Full data set in original file structure.
    train_set, train_files, train_labels,  test_set, test_files  =  dl.load_data(path)


    ####################
    # Find Input Shape #
    ####################
    # Take sample image
    img = cv2.imread(train_set.iloc[0][0])

    img_height = img.shape[0]
    img_width = img.shape[1]
    img_channels = img.shape[2]
    img_shape = (img_height, img_width, img_channels)
    img_size  = (img_height, img_width)

    # Any images that do not have this size will be cropped.
    # See source of train_model().
    # (What about if they're too small?


    ##################
    # Breed Analyzer #
    ##################

    # Initialize model
    model = a2c.mobilenet_v2_a(img_shape)

    # Prepare for training
    batch_size = 4
    epochs = 50
    n_fold = 5
    histories = []

    # (Find) run number
    run = 1
    while os.path.isdir(f'./chkpts/{model.name}/{breeder_version}/{data_folder}/Run_{run:02d}'):
        run += 1

    os.makedirs(f'./outputs/{model.name}/{breeder_version}/{data_folder}',exist_ok=True)
    file = open(f'./outputs/{model.name}/{breeder_version}/{data_folder}/Run_{run:02d}_train_params.txt',"w")
    file.write(f'batch_size = {batch_size}\n'+
               f'epochs = {epochs}\n'+
               f'n_fold = {n_fold}\n')
    file.close()

    kf = KFold(n_splits=n_fold, shuffle=True)

    # Train model: compile (configure for training), train, test, save
    histories, test_pred = a2c.train_model(model, batch_size, epochs, img_size,
                                       train_set, train_labels, test_files,
                                       n_fold, kf, breeder_version, data_folder,
                                       run)

    test_set['abnormality_pred'] = test_pred
    test_set.to_csv(f'./outputs/{model.name}/{breeder_version}/{data_folder}/output_Run_{run:02d}.csv', index=None)

    os.makedirs(f'./for_plots/{model.name}/{breeder_version}',exist_ok=True)
    f = open(f'for_plots/{model.name}/{breeder_version}/histories_Run_{run:02d}.pckl', 'wb')
    pickle.dump(f'for_plots/{model.name}/{breeder_version}/hist.histories', f)
    f.close()

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    file = open(f'./outputs/{model.name}/{breeder_version}/{data_folder}/Run_{run:02d}_time.txt',"w")
    file.write(f'Run train-and-test time (duration) = {elapsed.days} days, {elapsed.seconds} seconds, {elapsed.microseconds} microseconds.\n')
    file.close()


if __name__ == '__main__':
    main()

