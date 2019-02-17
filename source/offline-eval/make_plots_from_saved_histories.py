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
    data_category = '1-pre-processed'
    #data_category = '2-processed'
    eval_base = eval_root + data_category + '/'
    os.makedirs(eval_base, exist_ok=True)

    # Input locations:
    # Manually add the desired files to utilize.
    histories_paths = []
    histories_paths.append( {'name': 'MNv2a_Cr3', 'filepath' :
        '../output/mobilenet_v2_a/breeder_01/data_C/Run_03/' +
        'for_plots/histories_Run_03.pckl'} )
    histories_paths.append( {'name': 'MNv2a_Dr3', 'filepath' :
        '../output/mobilenet_v2_a/breeder_01/data_D/Run_03/' +
        'for_plots/histories_Run_03.pckl'} )

    for run in histories_paths:
        # Read input.
        print(f'Extracting histories from a run\'s pickle file...')
        file = open(run['filepath'], 'rb')
        histories = pickle.load(file) # histories for a single run
        file.close()

        # Set ultimate output location.
        plot_run_name = run['name']
        eval_fig_path = eval_base + f'{plot_run_name}/figures/'
        os.makedirs(eval_fig_path, exist_ok=True)

        # Make plots.
        print(f'Creating plots from run {plot_run_name} histories...')
        eval_figs.make_acc_loss_plots(histories,
                                      eval_fig_path, plot_run_name)


if __name__ == '__main__':
        main()


