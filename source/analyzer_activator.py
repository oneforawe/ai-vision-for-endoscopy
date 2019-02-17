#!/usr/bin/env python3
# filename: analyzer_activator.py

import os
import cv2
import data_loader as dl
import analyzers_2_categories as a2c
from keras.models import load_model
import model_evaluation as m_eval
import eval_figures as eval_figs
import datetime
import timer


def main():
    start_time = datetime.datetime.now()

    #############
    # Load Data #
    #############

    data_root = '../input-data/'
    data_category = '1-pre-processed'
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


    #####################
    # Activate Analyzer #
    #####################

    # Load saved (trained) model from a particular source (run)
    source_run_path = \
        '../output/train/MNv2a/1-pre-processed/by_abnorm/data_C/Run_01/'
    #   '../output/'
    model_short_name = \
        'MNv2a'
    #   '...'
    source_run_file_path = source_run_path + f'chkpts/ModelWhole_trained.hdf5'
    model = load_model(source_run_file_path)

    # Output location
    #output_root = '../output/test/infer/'
    output_root = '../output/infer/'
    output_base = output_root + f'{model_short_name}/' + \
                                f'{data_category}/{class_split}/{data_name}/'

    # Prepare for inference (done in batches)
    batch_size = 4  # C
    #batch_size = 20 # D (mobilenet)
    #batch_size = 10 # D (xception) #Got error (fixed by reducing to 10)
    #batch_size = 50 # F

    # Find this round's number ("inference round #..")
    rnd = 1
    while os.path.isfile(output_base +
                         f'Round_{rnd:02d}/duration_total_activator.txt'):
        rnd += 1
    rnd_path = output_base + f'Round_{rnd:02d}/'
    os.makedirs(rnd_path, exist_ok=True)

    # Apply trained model: infer
    inf_start_time = datetime.datetime.now()
    test_pred = a2c.apply_model(model, batch_size, img_size, test_files, rnd)
    inf_end_time = datetime.datetime.now()

    print('Now recording inference duration.')
    # Total:
    description = 'Round inference time (duration)'
    filepath = rnd_path + f'duration_inference.txt'
    inf_elapsed = inf_end_time - inf_start_time
    inf_tot_sec = timer.record_duration(inf_elapsed, description, filepath)
    # Per image:
    description = 'Round inference time (average duration per image)'
    filepath2 = rnd_path + f'duration_inference_per_img.txt'
    tot_num_imgs = len(test_set)
    inf_elapsed_per_img = inf_tot_sec / tot_num_imgs
    inf_imgs_per_sec =  tot_num_imgs / inf_tot_sec
    file = open(filepath, 'w')
    file.write(f'{description}\n = ' +
               f'{inf_elapsed_per_img} seconds/image.' +
               f'\n' +
               f'{inf_imgs_per_sec} images/second.\n')
    file.close()


    #############################
    # Save/Generate More Output #
    #############################

    print('Now saving inference output.')
    rnd_results_path = rnd_path + f'results/'
    os.makedirs(rnd_results_path, exist_ok=True)
    # output:
    test_set['abnormality_pred'] = test_pred
    rnd_results_file_path = rnd_results_path + f'output_scores.csv'
    test_set.to_csv(rnd_results_file_path, index=None)

    print('Now generating and saving evaluations and figures.')
    eval_path = rnd_path + f'evaluations/'
    eval_fig_path = eval_path + f'figures/'
    os.makedirs(eval_fig_path, exist_ok=True)
    # ROC fig
    plot_rnd_name = model_short_name + data_short_name + 'r' + str(rnd)
    eval_figs.make_roc_plot(test_set, eval_fig_path, plot_rnd_name)
    # evaluations data
    # (precision/recall, sensitivity/specificity, ROC/thresholds, etc)
    test_w_reckoning_choices, evaluations \
        = m_eval.make_eval_data(test_set, eval_path, plot_rnd_name)
    # thresh, CM fig, and reckonings
    eval_figs.pick_thresh_make_figures(evaluations, test_w_reckoning_choices,
                                       eval_path, eval_fig_path, plot_rnd_name)
    # (could show points on ROC curve for chosen threshold(s))

    print('Now recording total activator duration.')
    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    description = 'Total activator (load-infer-evaluate-etc) time (duration)'
    filepath = rnd_path + f'duration_total_activator.txt'
    timer.record_duration(elapsed, description, filepath)


if __name__ == '__main__':
    main()

