#!/usr/bin/env python3
# filename: training_figures.py

from sklearn.metrics import confusion_matrix


def make_acc_loss_plots(histories):

    # Want to plot accuracy and loss over training epochs
    plot_these = [['acc','val_acc','accuracy','Acc'],
                  ['loss','val_loss','loss','Loss']]
    # [(history key), (history key), (plot label), (filename affix)]
    for plot_this in plot_these:
        for i in range(len(histories)):
            plt.plot(histories[i].history[plot_this[0]],
                     linestyle='solid') # accuracy
        for i in range(len(histories)):
            plt.plot(histories[i].history[plot_this[1]],
                     linestyle='dashed') # loss
        plt.title('{} model {}'.format(
            histories_paths[output_index]['name'], plot_this[2] ) )
        plt.ylabel(plot_this[2])
        plt.xlabel('epoch')
        for i in len(histories):
            legend_labels.append(f'train fold {i}')
        for i in len(histories):
            legend_labels.append(f'validation f{i}')
        if plot_this[0]=='acc':
            plt.legend(legend_labels, loc='lower right')
        if plot_this[0]=='loss':
            plt.legend(legend_labels, loc='upper right')
        plt.legend()
        savefig("figures/{}_{}.png".format(
            histories_paths[output_index]['name'], plot_this[3]),
            dpi=300, bbox_inches='tight')

#def make_confusion_matrix_figure()
