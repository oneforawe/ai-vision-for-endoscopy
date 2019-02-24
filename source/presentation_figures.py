#!/usr/bin/env python3
# filename: presentation_figures.py

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pylab import savefig
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools


def make_acc_loss_plots(histories, eval_fig_path, plot_run_name):

    # Want to plot accuracy and loss over training epochs
    plot_these = [['acc','val_acc','accuracy','Acc'],
                  ['loss','val_loss','loss','Loss']]
    # [(history key), (history key), (plot label), (filename affix)]
    for plot_this in plot_these:
        plt.figure()
        plt.rcParams.update({'font.size': 16})
        for i in range(len(histories)):
            plt.plot(histories[i].history[plot_this[0]],
                     linestyle='solid') # accuracy
        for i in range(len(histories)):
            #plt.plot(histories[i].history[plot_this[1]],
            #         linestyle='dashed') # loss
            if i==0 or i == 3:
                plt.plot(histories[i].history[plot_this[1]],
                         linestyle='dashed') # loss
        plt.title(f'model {plot_this[2]}')
        plt.ylabel(f'{plot_this[2]}')
        plt.xlabel(f'epoch')
        legend_labels = []
        for i in range(len(histories)):
            legend_labels.append(f'train fold {i+1}')
        for i in range(len(histories)):
            #legend_labels.append(f'validation f{i+1}')
            if i==0 or i == 3:
                legend_labels.append(f'val. fold {i+1}')
        if plot_this[0]=='acc':
            plt.legend(legend_labels, loc='lower right')
        if plot_this[0]=='loss':
            plt.legend(legend_labels, loc='upper right')
        savefig(eval_fig_path +
                '{}_{}.png'.format(plot_run_name, plot_this[3]),
                dpi=300, bbox_inches='tight')
        plt.close()


