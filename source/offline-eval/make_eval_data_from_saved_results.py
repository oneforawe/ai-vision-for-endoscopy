#!/usr/bin/env python3
# filename: make_eval_data_from_saved_results.py

# Run in `source` folder with the following command:
# python -m offline-eval.make_eval_data_from_saved_results

import os
import pandas as pd
import model_evaluation as m_eval


def main():
    # Input:
    # Manually add the desired files to utilize.
    results_paths = []
    results_paths.append( {'name': 'MNv2a_Cr3', 'filepath' :
        '../output/mobilenet_v2_a/breeder_01/data_C/Run_03/' +
        'results/output_scores.csv'} )
    results_paths.append( {'name': 'MNv2a_Dr3', 'filepath' :
        '../output/mobilenet_v2_a/breeder_01/data_D/Run_03/' +
        'results/output_scores.csv'} )

    # Output location
    eval_root = '../output/offline-eval/'
    os.makedirs(eval_root,exist_ok=True)

    for run in results_paths:
        print(f'Reading scores (etc) from a run\'s output file...')
        test_set = pd.read_csv(run['filepath']) # dataframe

        plot_run_name = run['name']

        eval_path = eval_root+f'{plot_run_name}/'
        os.makedirs(eval_path,exist_ok=True)

        print(f'Creating evaluation data from run {plot_run_name} output...')
        m_eval.make_eval_data(test_set, eval_path, plot_run_name)


if __name__ == '__main__':
        main()


