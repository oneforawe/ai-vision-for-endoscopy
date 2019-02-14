#!/usr/bin/env python3
# filename: eval_figures.py

from matplotlib import pyplot as plt
from sklearn import metrics
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import confusion_matrix
from pylab import savefig
import numpy as np


def make_acc_loss_plots(histories, eval_fig_path, plot_run_name):

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
            plot_run_name, plot_this[2] ) )
        plt.ylabel(plot_this[2])
        plt.xlabel('epoch')
        legend_labels = []
        for i in range(len(histories)):
            legend_labels.append(f'train fold {i}')
        for i in range(len(histories)):
            legend_labels.append(f'validation f{i}')
        if plot_this[0]=='acc':
            plt.legend(legend_labels, loc='lower right')
        if plot_this[0]=='loss':
            plt.legend(legend_labels, loc='upper right')
        plt.legend()
        savefig(eval_fig_path +
                "{}_{}.png".format(plot_run_name, plot_this[3]),
                dpi=300, bbox_inches='tight')


def make_roc_plot(test_set, eval_fig_path, plot_run_name):
    roc_data = metrics.roc_curve(test_set['abnormality'],
                                        test_set['abnormality_pred'])
    fpr, tpr, thrsh = roc_data
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{plot_run_name} ROC curve')
    plt.legend(loc="lower right")
    savefig(eval_fig_path
            +f'{plot_run_name}_ROC_Curve.png',
            dpi=300, bbox_inches='tight')


def make_eval_metric_figures(evaluations, eval_path):

    # code from:
    # https://scikit-learn.org/
    # stable/auto_examples/model_selection/plot_confusion_matrix.html
    # #sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    #print(__doc__)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix (not normalized)')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          normalize=True,
                          title='Normalized confusion matrix')

    #plt.show()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    print(type(cm))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


