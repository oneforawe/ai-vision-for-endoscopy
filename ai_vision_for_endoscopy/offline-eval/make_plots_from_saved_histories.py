#!/usr/bin/env python3
# filename: make_plots_from_saved_histories.py

# Run in `source` folder with the following command:
# python -m offline-eval.make_plots_from_saved_histories

import os
import pickle
import eval_figures as eval_figs


def main():
    # Output location (root):
    eval_root = '../output/offline-eval/'
    #data_category = '1-pre-processed'
    data_category = '2-processed'
    class_split = 'by_abnorm'
    #class_split = 'by_region'

    # Input locations:
    # Manually add the desired files to utilize.
    histories_paths = list()
    histories_paths.append( {'model_short_name' : 'MNv2a',
        'data_short_name' : 'C',
        'run' : 1,
        'filepath' :
        '../output/train/MNv2a/2-processed/by_abnorm/data_C/Run_01/' +
        'histories/histories_Run_01.pckl'} )
    histories_paths.append( {'model_short_name' : 'MNv2a',
        'data_short_name' : 'C',
        'run' : 2,
        'filepath' :
        '../output/train/MNv2a/2-processed/by_abnorm/data_C/Run_02/' +
        'histories/histories_Run_02.pckl'} )
    histories_paths.append( {'model_short_name' : 'MNv2a',
        'data_short_name' : 'D',
        'run' : 1,
        'filepath' :
        '../output/train/MNv2a/2-processed/by_abnorm/data_D/Run_01/' +
        'histories/histories_Run_01.pckl'} )
    histories_paths.append( {'model_short_name' : 'MNv2a',
        'data_short_name' : 'E',
        'run' : 1,
        'filepath' :
        '../output/train/MNv2a/2-processed/by_abnorm/data_E/Run_01/' +
        'histories/histories_Run_01.pckl'} )
    histories_paths.append( {'model_short_name' : 'Xcp_a',
        'data_short_name' : 'C',
        'run' : 1,
        'filepath' :
        '../output/train/Xcp_a/2-processed/by_abnorm/data_C/Run_01/' +
        'histories/histories_Run_01.pckl'} )


    for run in histories_paths:
        # Read input.
        print(f'Extracting histories from a run\'s pickle file...')
        file = open(run['filepath'], 'rb')
        histories = pickle.load(file) # histories for a single run
        file.close()

        # Set ultimate output location.
        mn = run['model_short_name']
        dn = run['data_short_name']
        r  = run['run']
        plot_run_name = f'{mn}{dn}r{r}'
        eval_path = eval_root + f'{mn}/{data_category}/' + \
                    f'{class_split}/data_{dn}/Run_{r:02d}/evaluations/'
        eval_fig_path = eval_path + 'figures/'
        os.makedirs(eval_fig_path, exist_ok=True)

        # Make plots.
        print(f'Creating plots from run {plot_run_name} histories...')
        eval_figs.make_acc_loss_plots(histories,
                                      eval_fig_path, plot_run_name)


if __name__ == '__main__':
        main()


