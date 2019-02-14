#!/usr/bin/env python3
# filename: make_plots_from_saved_histories.py

import pickle
import eval_figures as eval_figs


def main():
    # Input:
    # Manually add the desired files to utilize.
    histories_paths = []
    histories_paths.append( {'name': 'MNv2a_Cr3', 'filepath' :
        '../output/mobilenet_v2_a/breeder_01/data_C/Run_03/' +
        'for_plots/histories_Run_03.pckl'} )
    histories_paths.append( {'name': 'MNv2a_Dr3', 'filepath' :
        '../output/mobilenet_v2_a/breeder_01/data_D/Run_03/' +
        'for_plots/histories_Run_03.pckl'} )

    # Output location
    eval_fig_path = '../output/offline-figures/'

    for run in histories_paths:
        print(f'Extracting histories from a run\'s pickle file...')
        file = open(run['filepath'],'rb')
        histories = pickle.load(file) # histories for a single run
        file.close()

        plot_run_name = run['name']

        print(f'Creating plots from run {plot_run_name} histories...')
        eval_figs.make_acc_loss_plots(histories,
                                      eval_fig_path, plot_run_name)


if __name__ == '__main__':
        main()


