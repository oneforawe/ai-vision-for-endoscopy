#!/usr/bin/env python3
# filename: make_cm_figs_from_saved_eval_data.py

# Run in `source` folder with the following command:
# python -m offline-eval.make_cm_figs_from_saved_eval_data

import os
import pandas as pd
import eval_figures as eval_figs


def main():
    # Output location (root):
    eval_root = '../output/offline-eval/'
    data_category = '1-pre-processed'
    #data_category = '2-processed'
    class_split = 'by_abnorm'
    #class_split = 'by_region'

    # Input locations:
    # Manually add the desired files to utilize.
    eval_data_paths = []
    eval_data_paths.append( {'model_short_name' : 'MNv2a',
        'data_short_name' : 'C',
        'run' : 1,
        'filepath' :
        '../output/offline-eval/MNv2a/2-processed/by_abnorm/data_C/Run_01/' +
        'evaluations/MNv2aCr1_eval_metrics.csv'} )
    eval_data_paths.append( {'model_short_name' : 'MNv2a',
        'data_short_name' : 'C',
        'run' : 2,
        'filepath' :
        '../output/offline-eval/MNv2a/2-processed/by_abnorm/data_C/Run_02/' +
        'evaluations/MNv2aCr2_eval_metrics.csv'} )
    eval_data_paths.append( {'model_short_name' : 'MNv2a',
        'data_short_name' : 'D',
        'run' : 1,
        'filepath' :
        '../output/offline-eval/MNv2a/2-processed/by_abnorm/data_D/Run_01/' +
        'evaluations/MNv2aDr1_eval_metrics.csv'} )
    eval_data_paths.append( {'model_short_name' : 'MNv2a',
        'data_short_name' : 'E',
        'run' : 1,
        'filepath' :
        '../output/offline-eval/MNv2a/2-processed/by_abnorm/data_E/Run_01/' +
        'evaluations/MNv2aEr1_eval_metrics.csv'} )
    eval_data_paths.append( {'model_short_name' : 'Xcp_a',
        'data_short_name' : 'C',
        'run' : 1,
        'filepath' :
        '../output/offline-eval/Xcp_a/2-processed/by_abnorm/data_C/Run_01/' +
        'evaluations/Xcp_aCr1_eval_metrics.csv'} )

    reckoning_paths = []
    reckoning_paths.append( {'filepath' :
        '../output/offline-eval/MNv2a/2-processed/by_abnorm/data_C/Run_01/'+
        'evaluations/MNv2aCr1_test_w_reckoning_choices.csv'} )
    reckoning_paths.append( {'filepath' :
        '../output/offline-eval/MNv2a/2-processed/by_abnorm/data_C/Run_02/'+
        'evaluations/MNv2aCr2_test_w_reckoning_choices.csv'} )
    reckoning_paths.append( {'filepath' :
        '../output/offline-eval/MNv2a/2-processed/by_abnorm/data_D/Run_01/'+
        'evaluations/MNv2aDr1_test_w_reckoning_choices.csv'} )
    reckoning_paths.append( {'filepath' :
        '../output/offline-eval/MNv2a/2-processed/by_abnorm/data_E/Run_01/'+
        'evaluations/MNv2aEr1_test_w_reckoning_choices.csv'} )
    reckoning_paths.append( {'filepath' :
        '../output/offline-eval/Xcp_a/2-processed/by_abnorm/data_C/Run_01/'+
        'evaluations/Xcp_aCr1_test_w_reckoning_choices.csv'} )


    for run_eval, run_reckon in zip(eval_data_paths, reckoning_paths):
        # Read input.
        print(f'Reading scores (etc) from a run\'s output file...')
        test_w_reckoning_choices = pd.read_csv(run_reckon['filepath'])
        evaluations = pd.read_csv(run_eval['filepath'])

        # Set ultimate output location.
        mn = run_eval['model_short_name']
        dn = run_eval['data_short_name']
        r  = run_eval['run']
        plot_run_name = f'{mn}{dn}r{r}'
        eval_path = eval_root + f'{mn}/{data_category}/' + \
                    f'{class_split}/data_{dn}/Run_{r:02d}/evaluations/'
        eval_fig_path = eval_path + 'figures/'
        os.makedirs(eval_fig_path, exist_ok=True)

        # Make plots and data.  (thresh, CM figs, and reckonings)
        eval_figs.pick_thresh_make_figures(evaluations,
                                           test_w_reckoning_choices,
                                           eval_path, eval_fig_path,
                                           plot_run_name)


if __name__ == '__main__':
        main()


