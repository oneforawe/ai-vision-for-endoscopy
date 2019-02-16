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
    reckoning_paths = []
    reckoning_paths.append( {'name': 'MNv2a_Cr3', 'filepath' :
        '../output/offline-eval/MNv2a_Cr3/' +
        'MNv2a_Cr3_test_w_reckoning_choices.csv'} )
    reckoning_paths.append( {'name': 'MNv2a_Dr3', 'filepath' :
        '../output/offline-eval/MNv2a_Dr3/' +
        'MNv2a_Dr3_test_w_reckoning_choices.csv'} )

    eval_data_paths = []
    eval_data_paths.append( {'name': 'MNv2a_Cr3', 'filepath' :
        '../output/offline-eval/MNv2a_Cr3/' +
        'MNv2a_Cr3_eval_metrics.csv'} )
    eval_data_paths.append( {'name': 'MNv2a_Dr3', 'filepath' :
        '../output/offline-eval/MNv2a_Dr3/' +
        'MNv2a_Dr3_eval_metrics.csv'} )

    # Output location
    eval_root = '../output/offline-eval/'
    os.makedirs(eval_root, exist_ok=True)

    for run_eval, run_reckon in zip(eval_data_paths, reckoning_paths):
        print(f'Reading scores (etc) from a run\'s output file...')
        test_w_reckoning_choices = pd.read_csv(run_reckon['filepath'])
        evaluations = pd.read_csv(run_eval['filepath'])

        # thresh, CM fig, and reckonings
        m_eval.pick_thresh_make_figures(evaluations, test_w_reckonings)


