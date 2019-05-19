In this folder are the source-code files for the project.

## Model Control
The main files:

|  file                       | purpose |
| --------------------------- | ------- |
| `analyzers_2_categories.py` | defines image-analyzer models and training and inference processes |
| `analyzer_breeder.py`       | run this script to breed (ie, create and train) models |
| `analyzer_activator.py`     | run this script to activate models for inference / image analysis |

The models are binary image classifiers, putting each image into one of two categories ("normal" and "abnormal"), hence the presence of `_2_categories` in the model-defining file.

Edit the code to select the desired model or define new models, tune parameters for training, reset the output base-directory, or otherwise change the behavior of the code.

## Model Evaluation
The main files for automatic model evaluation:

|  file                 | purpose |
| --------------------- | ------- |
| `model_evaluation.py` | calculates various metrics for model performance, finds good binary-score threshold for ROC curve |
| `eval_figures.py`     | plots accuracy and loss over training, ROC curves, confusion matrices |

