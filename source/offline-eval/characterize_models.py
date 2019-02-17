#!/usr/bin/env python3
# filename: characterize_models.py

# Run in `source` folder with the following command:
# python -m offline-eval.characterize_models

import os
import cv2
import pandas as pd
import analyzers_2_categories as a2c
#import keras
from contextlib import redirect_stdout


def main():
    # Output location (root):
    eval_root = '../output/offline-eval/'

    # Models to characterize:
    model_funcs = [a2c.mobilenet_v2_a,
                   a2c.mobilenet_v2_b,
                   a2c.mobilenet_v2_c,
                   a2c.xception_a,
                   a2c.xception_b]

    # A model depends on the type of input, and the proccessed data (images) is
    # different from the pre-processed data.
    data_types = ['1-pre-processed', '2-processed']

    for data_type in data_types:
        print(f'\nFOR data_type = {data_type}')

        # Set up model img_shape.
        data_location = f'../input-data/{data_type}'
        sample_img_file = get_sample_img_file(data_location)
        print(f'GIVEN sample_img_file = {sample_img_file}\n')
        sample_img = cv2.imread(sample_img_file)
        img_shape = sample_img.shape

        for model_func in model_funcs:
            # Set up model.
            print(f'SETTING UP / LOADING A MODEL AND IT\'S BASE.')
            model, model_short_name, base_model_name \
                = model_func(img_shape)
            base_model = model.get_layer(base_model_name)
            print(f'DONE SETTING UP / LOADING.')

            # Set ultimate output location.
            eval_base = eval_root + data_type + f'/{model_short_name}/'
            os.makedirs(eval_base, exist_ok=True)
            file_name = 'model_summary.txt'
            file_path = eval_base + file_name

            # Make characterization.
            print(f'-- Saving model summary for ' +
                  f'{model_short_name} with {data_type}...\n')
            print_to_file(file_path)(f'Overall Model Summary:\n')
            print_to_file(file_path)(f'model name:  {model.name}')
            print_to_file(file_path)(f'short name:  {model_short_name}')
            print_to_file(file_path)(f'# of layers: ' +
                                     f'{len(model.layers)}\n')
            model.summary(print_fn=print_to_file(file_path))
            print_to_file(file_path)(f'\n\n')
            print_to_file(file_path)(f'Base Model Summary:\n')
            print_to_file(file_path)(f'model name:  {base_model.name}')
            print_to_file(file_path)(f'# of layers: ' +
                                     f'{len(base_model.layers)}\n')
            base_model.summary(print_fn=print_to_file(file_path))



def get_sample_img_file(path):
    breaker = False
    for dirPath, subDirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg"):
                sample_file = os.path.join(dirPath, file)
                break
        if breaker == True:
            break
    return sample_file


def print_to_file(path):
    def print_string(s):
        with open(path,'a') as f:
            print(s, file=f)
    return print_string


if __name__ == '__main__':
        main()


