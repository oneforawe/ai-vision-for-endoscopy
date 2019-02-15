#!/usr/bin/env python3
# filename: make_cm_fig_from_saved_eval_data.py

# Run in `source` folder with the following command:
# python -m offline-eval.make_cm_fig_from_saved_eval_data

import os
import pandas as pd
import eval_figures as eval_figs
import model_evaluation as m_eval


def main():
    # Input:
    # Manually add the desired files to utilize.
    eval_data_paths = []
    eval_data_paths.append( {'name': 'MNv2a_Cr3', 'filepath' :
        '../output/offline-eval/MNv2a_Cr3/' +
        'MNv2a_Cr3_eval_metrics.csv'} )
    eval_data_paths.append( {'name': 'MNv2a_Dr3', 'filepath' :
        '../output/offline-eval/MNv2a_Dr3/' +
        'MNv2a_Dr3_eval_metrics.csv'} )

    reckoning_paths = []
    reckoning_paths.append( {'name': 'MNv2a_Cr3', 'filepath' :
        '../output/offline-eval/MNv2a_Cr3/' +
        'MNv2a_Cr3_test_w_reckoning_choices.csv'} )
    reckoning_paths.append( {'name': 'MNv2a_Dr3', 'filepath' :
        '../output/offline-eval/MNv2a_Dr3/' +
        'MNv2a_Dr3_test_w_reckoning_choices.csv'} )

    # Output location
    eval_root = '../output/offline-eval/'
    os.makedirs(eval_root,exist_ok=True)

    for run_eval, run_reckon in zip(eval_data_paths, reckoning_paths):
        print(f'Reading scores (etc) from a run\'s output file...')
        evaluations = pd.read_csv(run_eval['filepath'])
        test_w_reckoning_choices = pd.read_csv(run_reckon['filepath'])

        thresh = m_eval.pick_threshold(evaluations)
        test_w_reckonings \
            = test_w_reckoning_choices[['abnormality',
                                        'abnormality_pred',
                                        f'{thresh:0.3f}']]
        plot_run_name = run_eval['name']
        eval_fig_path = f'../output/' + \
                        f'offline-eval/{plot_run_name}/figures/'
        os.makedirs(eval_fig_path,exist_ok=True)

        print(f'Creating confusion matrix figures from run ' +
              f'{plot_run_name} eval data...')
        eval_figs.make_eval_metric_figures(test_w_reckonings, thresh,
                                           eval_fig_path, plot_run_name)


if __name__ == '__main__':
        main()


