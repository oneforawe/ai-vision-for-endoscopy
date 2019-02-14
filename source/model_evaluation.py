#!/usr/bin/env python3
# filename: model_evaluation.py

"""
model_evaluation
~~~~~~~~~~~~
A library for generating plots of model evaluation metrics while
training/testing and for previous train/test sessions.
"""
# See below for vocabulary/variable definitions.

import os
import pandas as pd
import numpy as np


def make_eval_data(test_set, eval_path):
    # for each threshold, there is a confusion matrix with TP,FP,TN,FN
    # from which TPR = TP/P and FPR = FP/N can be calculated
    test_w_reckonings = test_set[['abnormality',
                                  'abnormality_pred']].copy()
    evaluations = pd.DataFrame()
    # including FNs, FPs, confusion matrices, ROC curve points

    divisions = 1001
    for step in range(divisions+1):
        thrsh = np.linspace(0,1.001,divisions+1)[step]
        TP = 0; FP = 0; TN = 0; FN = 0
        reckonings = []
        for i in range(len(test_set)):
            label = test_set['abnormality'].loc[i]
            pred  = test_set['abnormality_pred'].loc[i]
            reckoning = 0 if pred < thrsh else 1
            if reckoning == label:
                if reckoning == 1:
                    TP += 1
                else:
                    TN += 1
            if reckoning != label:
                if reckoning == 1:
                    FP += 1
                else:
                    FN += 1
            reckonings.append(reckoning)
        confusion_matrix = [[TP, FP], [FN, TN]]
        MP = TP+FP
        MN = TN+FN
        P  = TP+FN
        N  = TN+FP
        if MP==0:
            PPV = None
        else:
            PPV = TP/MP
        TPR = TP/P
        TNR = TN/N
        FPR = FP/N
        FNR = FN/P
    test_w_reckonings[f'{thrsh:0.3f}'] = reckonings
    eval_thrsh = [round(thrsh, 3), FN, FP,
                  confusion_matrix, PPV, TPR, TNR, FPR, FNR]
    evaluations[f'{thrsh:0.3f}'] = eval_thrsh

    # Save to csv
    evaluations.to_csv(eval_path+f'eval_metrics.csv', index=None)
    test_w_reckonings.to_csv(eval_path +
                             f'test_w_reckonings.csv', index=None)

    return test_w_reckonings, evaluations


# Vocabulary / Variables
# ======================
#
# mp  = measured positive (a test/measured case that gets a
#                          "positive" result from a binary test)
# mn  = measured negative (a test/measured case that gets a
#                          "negative" result from a binary test)
# p   = positive         (an actually positive test case; should get mp)
# n   = negative         (an actually negative test case; should get mn)
# tp  = "true positive"   (a mp that is a p; a correct-measurement case)
# tn  = "true negative"   (a mn that is a n; a correct-measurement case)
# fp  = "false positive"  (a mp that is a n; an incorrect-meas. case)
# fn  = "false negative"  (a mn that is a p; an incorrect-meas. case)
#
# MP  = Measured Positive (total number of mp)
# MN  = Measured Negative (total number of mn)
# P   = Positives         (total number of p)
# N   = Negatives         (total number of n)
# TP  = "True Positives"  (total number of tp)
# TN  = "True Negatives"  (total number of tn)
# FP  = "False Positives" (total number of fp)
# FN  = "False Negatives" (total number of fn)
#
# P  = TP+FN
# N  = TN+FP
# MP = TP+FP
# MN = TN+FN
#
# TP/MP = PPV = Precision                  Positive Predictive Value
# TP/P  = TPR = Sensitivity = Recall       True  P rate
# TN/N  = TNR = Specificity = Selectivity  True  N rate
# FP/N  = FPR = Fallout                    False P rate
# FN/P  = FNR = Miss Rate                  False P rate
#
# TPR = 1 - FNR        (Sensitivity|Recall) = 1-(Miss Rate)
# TNR = 1 - FPR   (Specificity|Selectivity) = 1-(Fallout)

