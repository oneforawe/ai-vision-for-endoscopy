#!/usr/bin/env python3
# filename: make_eval_data_from_saved_results.py

# Run in `source` folder with the following command:
# python -m offline-eval.make_eval_data_from_saved_results

import os
import pandas as pd
import model_evaluation as m_eval


def main():
    # Output location (root):
    eval_root = '../output/offline-eval/'
    #data_category = '1-pre-processed'
    data_category = '2-processed'
    class_split = 'by_abnorm'
    #class_split = 'by_region'

    # Input locations:
    # Manually add the desired files to utilize.
    results_paths = []
    results_paths.append( {'model_short_name' : 'MNv2a',
        'data_short_name' : 'C',
        'run' : 1,
        'filepath' :
        '../output/train/MNv2a/2-processed/by_abnorm/data_C/Run_01/' +
        'results/output_scores.csv'} )
    results_paths.append( {'model_short_name' : 'MNv2a',
        'data_short_name' : 'C',
        'run' : 2,
        'filepath' :
        '../output/train/MNv2a/2-processed/by_abnorm/data_C/Run_02/' +
        'results/output_scores.csv'} )
    results_paths.append( {'model_short_name' : 'MNv2a',
        'data_short_name' : 'D',
        'run' : 1,
        'filepath' :
        '../output/train/MNv2a/2-processed/by_abnorm/data_D/Run_01/' +
        'results/output_scores.csv'} )
    results_paths.append( {'model_short_name' : 'MNv2a',
        'data_short_name' : 'E',
        'run' : 1,
        'filepath' :
        '../output/train/MNv2a/2-processed/by_abnorm/data_E/Run_01/' +
        'results/output_scores.csv'} )
    results_paths.append( {'model_short_name' : 'Xcp_a',
        'data_short_name' : 'C',
        'run' : 1,
        'filepath' :
        '../output/train/Xcp_a/2-processed/by_abnorm/data_C/Run_01/' +
        'results/output_scores.csv'} )


    for run in results_paths:
        # Read input.
        print(f'Reading scores (etc) from a run\'s output file...')
        test_set = pd.read_csv(run['filepath']) # dataframe

        # Set ultimate output location.
        mn = run['model_short_name']
        dn = run['data_short_name']
        r  = run['run']
        plot_run_name = f'{mn}{dn}r{r}'
        eval_path = eval_root + f'{mn}/{data_category}/' + \
                    f'{class_split}/data_{dn}/Run_{r:02d}/evaluations/'
        os.makedirs(eval_path, exist_ok=True)

        # Make data.
        print(f'Creating evaluation data from run {plot_run_name} output...')
        m_eval.make_eval_data(test_set, eval_path, plot_run_name)


if __name__ == '__main__':
        main()


