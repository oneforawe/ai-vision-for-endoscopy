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
    #data_path = '../data/1-pre-processed/A'
    #data_name = "data_A"
    # B: 26 images
    #data_path = '../data/1-pre-processed/B'
    #data_name = "data_B"
    # C: 200 images (Abnormal=Blood)
    data_path = '../input-data/1-pre-processed/C'
    data_name = "data_C"
    # D: 2000 images
    #data_path = '../data/1-pre-processed/D'
    #data_name = "data_D"
    # Full data set in original file structure.
    train_set, train_files, train_labels,  test_set, test_files  =  dl.load_data(data_path)


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

    # Output location
    output_root = '../output/'
    output_base = output_root+f'{model.name}/{breeder_version}/{data_name}/'

    # Prepare for training
    batch_size = 4  # C
    #batch_size = 20 # D
    epochs = 50
    n_fold = 4 # C
    #n_fold = 5 # D
    histories = []

    # (Find) run number
    run = 1
    while os.path.isfile(output_base+f'/Run_{run:02d}/run_duration.txt'):
        run += 1
    run_path = output_base+f'/Run_{run:02d}/'

    os.makedirs(run_path,exist_ok=True)
    file = open(run_path+f'train_params.txt',"w")
    file.write(f'batch_size = {batch_size}\n'+
               f'epochs = {epochs}\n'+
               f'n_fold = {n_fold}\n')
    file.close()

    kf = KFold(n_splits=n_fold, shuffle=True)

    # Train model: compile (configure for training), train, test, save
    histories, test_pred = a2c.train_model(model, batch_size, epochs, img_size,
                                       train_set, train_labels, test_files,
                                       n_fold, kf, run_path, run)

    print("Now saving results.")
    test_set['abnormality_pred'] = test_pred
    os.makedirs(run_path+f'results',exist_ok=True)
    test_set.to_csv(run_path+f'results/output_scores.csv', index=None)

    os.makedirs(run_path+f'for_plots',exist_ok=True)
    f = open(run_path+f'for_plots/histories_Run_{run:02d}.pckl', 'wb')
    pickle.dump(run_path+f'for_plots/hist.histories', f)
    f.close()

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    days = elapsed.days
    secs = elapsed.seconds
    hrs  = secs//3600
    mins = (secs-hrs*3600)//60
    secs = secs-hrs*3600-mins*60
    file = open(run_path+f'run_duration.txt',"w")
    file.write(f'Run train-and-test time (duration) = '
               +f'{days} days, {hrs} hours, {mins} minutes, {secs} seconds, '
               +f'{elapsed.microseconds} microseconds.\n')
    file.close()


if __name__ == '__main__':
    main()

