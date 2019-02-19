#!/usr/bin/env python3
# filename: model_evaluation.py

"""
model_evaluation
~~~~~~~~~~~~~~~~
A library for generating plots of model evaluation metrics while
training/testing and for previous train/test sessions.
"""
# See below for vocabulary/variable definitions.

import os
import pandas as pd
import numpy as np


def make_eval_data(test_set, eval_path, plot_run_name):
    # We want to collect these (for each threshold):
    eval_names = ['Score Threshold',
                  'Measurements (M)',
                  'Confusion Matrix (cm)',
                  'True Pos.s (TP)', 'True Neg.s (TN)',
                  'False Pos.s (FP)', '*False Neg.s* (FN)',
                  'Positives (P)', 'Negatives (N)',
                  'Meas.Pos.s (MP)', 'Meas.Neg.s (MN)',
                  'Correct Meas.s (CM)', 'Incorrect Meas.s (IM)',
                  'Accuracy (ACC)',
                  'Precision (PPV)', 'Recall/Sensitivity (TPR)',
                  'Specificity/Selectivity (TNR)',
                  'Fallout (FPR)', 'Miss Rate (FNR)',
                  'F1 Score (F1)']
                  # including TPR and FPR as ROC curve points
    evaluations = pd.DataFrame(columns=eval_names)
    test_w_reckoning_choices = test_set.copy()

    # Run through many threshold values from 0 to 1 (plus another step).
    # (1000 steps from zero to one progress by 0.001, and additional last step
    #  ends at 1.001)
    thrsh_steps = 1001
    for step in range(thrsh_steps+1):
        thrsh = np.linspace(0,1.001,thrsh_steps+1)[step]
        TP = 0; FP = 0; TN = 0; FN = 0
        reckonings = []
        for j in range(len(test_set)):
            # j = image index
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

        # Calculations
        M  = TP+FP+TN+FN
        confusion_matrix = [[TP, FP], [FN, TN]]
        P  = TP+FN
        N  = TN+FP
        MP = TP+FP
        MN = TN+FN
        CM = TP+TN
        IM = FP+FN
        ACC = CM/M
        if MP==0:
            PPV = None
        else:
            PPV = TP/MP
        TPR = TP/P
        TNR = TN/N
        FPR = FP/N
        FNR = FN/P
        if PPV is None:
            F1 = None
        else:
            F1  = 2*PPV*TPR/(PPV+TPR)
        test_w_reckoning_choices[f'{thrsh:0.3f}'] = reckonings
        eval_values = [round(thrsh, 3), M,
                       confusion_matrix, TP, FN, FP, FN,
                       P, N, MP, MN, CM, IM,
                       ACC, PPV, TPR, TNR, FPR, FNR, F1]

        evaluations.loc[step] = eval_values

    # Save to csv
    evaluations.to_csv(eval_path
                       +f'{plot_run_name}_eval_metrics.csv', index=None)
    # A bit excessive to save (can re-enable when wanted):
    #test_w_reckoning_choices.to_csv(eval_path +
    #    f'{plot_run_name}_test_w_reckoning_choices.csv', index=None)

    return test_w_reckoning_choices, evaluations


def pick_threshold(evaluations, eval_path, plot_run_name):
    # The default is thrsh = 0.5, but we can (also) find thrsh values that
    # achieve a certain number of fn's (FN=0) and fp's (FP minimized).
    df1 = evaluations
    FN_min = df1['*False Neg.s* (FN)'].min()          # Ideally w/ FN = 0
    df2 = df1[df1['*False Neg.s* (FN)'] == FN_min]    # Thresholds w/ FN small
    FP_min = df2['False Pos.s (FP)'].min()            # and minimum FP, and
    df3 = df2.loc[df2['False Pos.s (FP)'] == FP_min]  # among these, find
    good_thrsh_min = df3['Score Threshold'].min() # the smallest threshold and
    good_thrsh_max = df3['Score Threshold'].max() # the largest threshold..
    # ..giving the range of good thresholds.
    print(f'Good thresholds are in the interval:')
    print(f'  [{good_thrsh_min}, {good_thrsh_max}]')
    thrsh_interval_filepath = eval_path + f'{plot_run_name}_eval_thresholds.txt'
    file = open(thrsh_interval_filepath, 'w')
    file.write(f'Good thresholds are in the interval:\n' +
               f'  [{good_thrsh_min}, {good_thrsh_max}]\n')
    file.close()
    # Among these, pick the closest to 0.5:
    if good_thrsh_min <= 0.5 <= good_thrsh_max:
        thrsh = 0.5
    elif 0.5 < good_thrsh_min:
        thrsh = good_thrsh_min
    elif 0.5 > good_thrsh_max:
        thrsh = good_thrsh_max

    return thrsh



# Vocabulary / Variables
# ======================
#
# threshold = a score boundary value, where scores below it indicate
#             one category/class and scores above it indicate the other
# reckoning = the category/class determined from a score and threshold
#
# m   = measurement       (a test/measured case)
# mp  = measured positive (a test/measured case that gets a
#                          "positive" result from a binary test)
# mn  = measured negative (a test/measured case that gets a
#                          "negative" result from a binary test)
# p   = positive          (an actually positive test case; should get mp)
# n   = negative          (an actually negative test case; should get mn)
# tp  = "true positive"   (a mp that is a p; a correct-measurement for mp)
# tn  = "true negative"   (a mn that is a n; a correct-measurement for mn)
# fp  = "false positive"  (a mp that is a n; an incorrect-meas. for mp)
# fn  = "false negative"  (a mn that is a p; an incorrect-meas. for mn)
# cm  = correct measurem. (a correct or "true" measurement; tp or tn)
# im  = incorrect meas.   (an incorrect or "false" meas.; fp or fn)
#
# M   = Measurements       (total number of m)
# MP  = Measured Positives (total number of mp)
# MN  = Measured Negatives (total number of mn)
# P   = Positives          (total number of p)
# N   = Negatives          (total number of n)
# TP  = "True Positives"   (total number of tp)
# TN  = "True Negatives"   (total number of tn)
# FP  = "False Positives"  (total number of fp)
# FN  = "False Negatives"  (total number of fn)
# CM  = Correct Measurem.  (total number of cm)
# IM  = Incorrect Meas.    (total number of im)
#
# M  = TP+TN+FP+FN
# MP = TP+FP
# MN = TN+FN
# CM = TP+TN
# IM = FP+FN
# P  = TP+FN
# N  = TN+FP
#
# CM/M  = ACC = Accuracy
# TP/MP = PPV = Precision                  Positive Predictive Value
# TP/P  = TPR = Sensitivity = Recall       True  P rate (1)
# TN/N  = TNR = Specificity = Selectivity  True  N rate (2)
# FP/N  = FPR = Fallout                    False P rate
# FN/P  = FNR = Miss Rate                  False P rate
# F1 Score = 2(PPV*TPR)/(PPV+TPR)
#   (1/F1) = (1/2)*[ (1/PPV) + (1/TPR) ]   (perhaps better to look at inverses)
#
# (1) Probability of Correct Detection for Positives
# (2) Probability of Correct Detection for Negatives
#
# TPR = 1 - FNR        (Sensitivity|Recall) = 1-(Miss Rate)
# TNR = 1 - FPR   (Specificity|Selectivity) = 1-(Fallout)
#
#
# Want to record
# 'Score Threshold'
# 'Measurements (M)',
# 'Positives (P)', 'Negatives (N)',
# 'Meas.Pos. (MP)', 'Meas.Neg. (MN)',
# 'True Pos. (TP)', 'True Neg. (TN)',
# 'False Negatives (FN)', 'False Positives (FP)',
# 'Confusion Matrix (cm)',
# 'Accuracy (ACC)',
# 'Precision (PPV)', 'Recall/Sensitivity (TPR)',
# 'Specificity/Selectivity (TNR)',
# 'Fallout (FPR)', 'Miss Rate (FNR)']
#
#
# Note:
# Could add
# MCC = Mathews Correlation Coefficient
# BM  = (Bookmaker) Informedness
# MK  = Markedness

