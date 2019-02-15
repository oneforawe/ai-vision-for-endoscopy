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
    # We want to collect these (for each threshold):
    eval_names = ['Score Threshold',
                  'False Negatives (FN)', 'False Positives (FP)',
                  'Confusion Matrix (cm)', 'Precision (PPV)',
                  'Recall/Sensitivity (TPR)',
                  'Specificity/Selectivity (TNR)',
                  'Fallout (FPR)', 'Miss Rate (FNR)']
                  # including TPR and FPR as ROC curve points
    evaluations = pd.DataFrame(columns=eval_names)
    test_w_reckoning_choices = test_set.copy()
    #test_w_reckoning_choices = test_set[['abnormality',
    #                                     'abnormality_pred']].copy()

    # run through many threshold values from 0 to 1 (plus another step)
    # (1000 steps from zero to one progress by 0.001, and additional
    #  last step ends at 1.001)
    i = 0
    thrsh_steps = 1001
    for step in range(thrsh_steps+1):
        thrsh = np.linspace(0,1.001,thrsh_steps+1)[step]
        TP = 0; FP = 0; TN = 0; FN = 0
        reckonings = []
        for j in range(len(test_set)):
            label = test_set['abnormality'].loc[j]
            pred  = test_set['abnormality_pred'].loc[j]
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
        test_w_reckoning_choices[f'{thrsh:0.3f}'] = reckonings
        eval_values = [round(thrsh, 3), FN, FP,
                       confusion_matrix, PPV, TPR, TNR, FPR, FNR]
        evaluations.loc[i] = eval_values
        i += 1

    # Save to csv
    evaluations.to_csv(eval_path+f'eval_metrics.csv', index=None)
    test_w_reckoning_choices.to_csv(eval_path +
        f'test_w_reckoning_choices.csv', index=None)

    return test_w_reckoning_choices, evaluations


def pick_threshold(test_w_reckoning_choices):
    # First default is thresh = 0.5

    df1 = test_w_reckoning
    df2 = df1[df1[1] == 0]
    minimum = df2[2].min()
    df3 = df2.loc[df2[2] == minimum]
    good_thrsh_min = df3[0].min()
    good_thrsh_max = df3[0].max()
    print(f'Good thresholds are in the interval:')
    print(f'  [{good_thrsh_min}, {good_thrsh_max}]')
    # Among these, pick the closest to 0.5...
    # or just set to 0.5 for now:
    thresh = 0.5

    return thresh



# Vocabulary / Variables
# ======================
#
# threshold = a score boundary value, where scores below it indicate
#             one category/class and scores above it indicate the other
# reckoning = the category/class determined from a score and threshold
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
# TP/P  = TPR = Sensitivity = Recall       True  P rate (1)
# TN/N  = TNR = Specificity = Selectivity  True  N rate (2)
# FP/N  = FPR = Fallout                    False P rate
# FN/P  = FNR = Miss Rate                  False P rate
#
# (1) Probability of Correct Detection for Positives
# (2) Probability of Correct Detection for Negatives
#
# TPR = 1 - FNR        (Sensitivity|Recall) = 1-(Miss Rate)
# TNR = 1 - FPR   (Specificity|Selectivity) = 1-(Fallout)

