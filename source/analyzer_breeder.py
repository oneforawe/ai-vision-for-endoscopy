#!/usr/bin/env python3
# filename: analyzer_breeder.py

import os
import cv2
import data_loader as dl
import analyzers_2_categories as a2c
from sklearn.model_selection import KFold
import model_evaluation as m_eval
import eval_figures as eval_figs
import datetime
import timer
import pickle


def main():
    start_time = datetime.datetime.now()

    #############
    # Load Data #
    #############

    data_root = '../input-data/'
    #data_category = '1-pre-processed'
    data_category = '2-processed'
    data_base = data_root + data_category + '/'
    # A: 138062 images. Full data set in original file structure.
    #data_path = data_base+'A'
    #data_name = 'data_A'
    #data_short_name = 'A'
    # B: 26 images.
    #data_path = data_base+'B'
    #data_name = 'data_B'
    #data_short_name = 'B'
    # C: 200 images. (Abnormal=Blood)
    #data_path = data_base+'C'
    #data_name = 'data_C'
    #data_short_name = 'C'
    # D: 2000 images.
    data_path = data_base+'D'
    data_name = 'data_D'
    data_short_name = 'D'
    # E: 10000 images.
    #data_path = data_base+'E'
    #data_name = 'data_E'
    #data_short_name = 'E'
    # F: 138062 images. Full data (minus vids) set in modified file structure.
    #data_path = data_base+'F'
    #data_name = 'data_F'
    #data_short_name = 'F'

    # Load:
    class_split = 'by_abnorm'
    #class_split = 'by_region'
    train_set, train_files, train_labels, \
           test_set, test_files \
        =  dl.load_data_2class_abnormality(data_path)
    #   =  dl.load_data_4class_region(data_path)


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
    model, model_short_name, base_model_name \
        = a2c.mobilenet_v2_d(img_shape)  # with deep "fine-tuning"
    # Options:
    #   = a2c.mobilenet_v2_a(img_shape)  # without "fine-tuning"
    #   = a2c.mobilenet_v2_b(img_shape)  # with shallow "fine-tuning"
    #   = a2c.mobilenet_v2_c(img_shape)  # with deep "fine-tuning"
    #   = a2c.mobilenet_v2_d(img_shape)  # with deep "fine-tuning"
    #   = a2c.xception_a(img_shape)      # without "fine-tuning"
    #   = a2c.xception_b(img_shape)      # with shallow "fine-tuning"

    # Output location
    #output_root = '../output/test/train/'
    #output_root = '../output/cpu/train/'
    output_root = '../output/train/'
    output_base = output_root + f'{model_short_name}/' + \
                                f'{data_category}/{class_split}/{data_name}/'

    # Prepare for training
    #batch_size = 4  # C
    batch_size = 20 # D (mobilenet)
    #batch_size = 10 # D (xception) #Got error (fixed by reducing to 10)
    #batch_size = 10 #E (mobilenet) #Got error at 40, reducing to 10
    #batch_size = 100 # F
    epochs = 50
    #n_fold = 4 # C
    n_fold = 5 # D,E,F
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

    print('Now recording train-and-test durations.')
    # Total:
    description = 'Run train-and-test time (duration)'
    filepath1 = run_path + f'duration_train_and_test.txt'
    tnt_elapsed = tnt_end_time - tnt_start_time
    tnt_tot_sec = timer.record_duration(tnt_elapsed, description, filepath1)
    # Per image:
    description = 'Run train-and-test time (average duration per image)'
    filepath2 = run_path + f'duration_train_and_test_per_img.txt'
    tot_num_imgs = len(train_set) + len(test_set)
    tnt_elapsed_per_img = tnt_tot_sec / tot_num_imgs
    tnt_imgs_per_sec =  tot_num_imgs / tnt_tot_sec
    file = open(filepath2, 'w')
    file.write(f'{description}\n = ' +
               f'({tnt_tot_sec} seconds) / ({tot_num_imgs} images)\n = ' +
               f'{tnt_elapsed_per_img} seconds/image.' +
               f'\n\n' +
               f'({tot_num_imgs} images) / ({tnt_tot_sec} seconds)\n = ' +
               f'{tnt_imgs_per_sec} images/second.\n')
    file.close()


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
    eval_figs.pick_thresh_make_figures(evaluations, test_w_reckoning_choices,
                                       eval_path, eval_fig_path, plot_run_name)
    # (could show points on ROC curve for chosen threshold(s))

    print('Now recording total breeder duration.')
    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    description = 'Total breeder (train-test-evaluate-etc) time (duration)'
    filepath3 = run_path + f'duration_total_breeder.txt'
    timer.record_duration(elapsed, description, filepath3)


if __name__ == '__main__':
    main()

