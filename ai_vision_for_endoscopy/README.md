In this folder are the source-code files for the project.

|  file                       | purpose                                                           |
| --------------------------- | ----------------------------------------------------------------- |
| `analyzers_2_categories.py` | defines image-analyzer models and training process                |
| `analyzer_breeder.py`       | run this script to breed (ie, create and train) models            |
| `analyzer_activator.py`     | run this script to activate models for inference / image analysis |

The models are binary image classifiers, putting each image into one of two categories ("normal" and "abnormal"), hence the presence of `_2_categories` in the model-defining file.
