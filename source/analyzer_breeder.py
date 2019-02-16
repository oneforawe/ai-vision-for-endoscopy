#!/usr/bin/env python3
# filename: analyzer_breeder.py

import os
import pickle
import cv2
import data_loader as dl
import analyzers_2_categories as a2c
from sklearn.model_selection import KFold
import model_evaluation as m_eval
import eval_figures as eval_figs
import timer


def main():
    start_time = datetime.datetime.now()

    #############
    # Load Data #
    #############

    data_root = '../input-data/'
    data_base = data_root+'1-pre-processed/'
    # A: 138062 images. Full data set in original file structure.
    #data_path = data_base+'A'
    #data_name = 'data_A'
    #data_short_name = 'A'
    # B: 26 images.
    #data_path = data_base+'B'
    #data_name = 'data_B'
    #data_short_name = 'B'
    # C: 200 images. (Abnormal=Blood)
    data_path = data_base+'C'
    data_name = 'data_C'
    data_short_name = 'C'
    # D: 2000 images.
    #data_path = data_base+'D'
    #data_name = 'data_D'
    #data_short_name = 'D'
    # F: 138062 images. Full data set in modified file structure.
    #data_path = data_base+'F'
    #data_name = 'data_F'
    #data_short_name = 'F'

    # Load:
    train_set, train_files, train_labels, \
           test_set, test_files  =  dl.load_data(data_path)


    ####################
    # Find Input Shape #
    ####################

    # Take sample image
    img = cv2.imread(train_set.iloc[0][0])

    img_height = img.shape[0]
    img_width  = img.shape[1]
    img_channels = img.shape[2]
    img_shape = (img_height, img_width, img_channels)
    img_size  = (img_height, img_width)

    # Any images that do not have this size will be cropped.
    # See source of train_model().
    # (What about if they're too small?)


    ##################
    # Breed Analyzer #
    ##################

    # Initialize model
    model, model_short_name \
        = a2c.mobilenet_v2_a(img_shape)  # without "fine-tuning"
    #   = a2c.mobilenet_v2_b(img_shape)  # with shallow "fine-tuning"
    #   = a2c.mobilenet_v2_c(img_shape)  # with deep "fine-tuning"
    #   = a2c.xception_a(img_shape)      # without "fine-tuning"
    #   = a2c.xception_a(img_shape)      # with shallow "fine-tuning"

    # Output location
    output_root = '../output/test/train/'
    #output_root = '../output/train/'
    output_base = output_root + f'{model.name}/{data_name}/'

    # Prepare for training
    batch_size = 4  # C
    #batch_size = 20 # D (mobilenet)
    #batch_size = 10 # D (xception) #Got error (fixed by reducing to 10)
    #batch_size = 50 # F
    epochs = 50
    n_fold = 4 # C
    #n_fold = 5 # D,F
    histories = []

    # Find this run's number ("train-and-test run #..")
    run = 1
    while os.path.isfile(output_base +
                         f'Run_{run:02d}/duration_total_breeder.txt'):
        run += 1
    run_path = output_base + f'Run_{run:02d}/'

    os.makedirs(run_path, exist_ok=True)
    file = open(run_path+f'train_params.txt', 'w')
    file.write(f'batch_size = {batch_size}\n' +
               f'epochs = {epochs}\n' +
               f'n_fold = {n_fold}\n')
    file.close()

    kf = KFold(n_splits=n_fold, shuffle=True)

    # Train model: compile (configure for training), train, test, save (& time)
    tnt_start_time = datetime.datetime.now()
    histories, test_pred = a2c.train_model(model, batch_size, epochs, img_size,
                                           train_set, train_labels, test_files,
                                           n_fold, kf, run_path, run)
    tnt_end_time = datetime.datetime.now()

    print('Now recording train-and-test duration.')
    tnt_elapsed = tnt_end_time - tnt_start_time
    description = 'Run train-and-test time (duration)'
    filepath = run_path + f'duration_train_and_test.txt'
    timer.record_duration(interval, description, filepath)


    #############################
    # Save/Generate More Output #
    #############################

    print('Now saving training output and histories.')
    run_results_path = run_path + f'results/'
    os.makedirs(run_results_path, exist_ok=True)
    # output:
    test_set['abnormality_pred'] = test_pred
    run_results_file_path = run_results_path + \
                            f'output_scores.csv'
    test_set.to_csv(run_results_file_path, index=None)
    # histories:
    run_histories_path = run_path + f'histories/'
    run_histories_file_path = run_histories_path + \
                              f'histories_Run_{run:02d}.pckl'
    os.makedirs(run_histories_path, exist_ok=True)
    f = open(run_histories_file_path, 'wb')
    pickle.dump(histories, f)
    f.close()

    print('Now generating and saving evaluations and figures.')
    eval_path = run_path + f'evaluations/'
    eval_fig_path = eval_path + f'figures/'
    os.makedirs(eval_fig_path, exist_ok=True)
    # histories
    plot_run_name = model_short_name + data_short_name + 'r' + str(run)
    eval_figs.make_acc_loss_plots(histories, eval_fig_path, plot_run_name)
    # ROC fig
    eval_figs.make_roc_plot(test_set, eval_fig_path, plot_run_name)
    # evaluations data
    # (precision/recall, sensitivity/specificity, ROC/thresholds, etc)
    test_w_reckoning_choices, evaluations \
        = m_eval.make_eval_data(test_set, eval_path, plot_run_name)
    # thresh, CM fig, and reckonings
    eval_figs.pick_thresh_make_figures(evaluations, test_w_reckonings)

    print('Now recording total breeder duration.')
    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    description = 'Total breeder (train-test-evaluate-etc) time (duration)'
    filepath = run_path + f'duration_total_breeder.txt'
    timer.record_duration(interval, description, filepath)


if __name__ == '__main__':
    main()

