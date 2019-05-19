# AI Vision for Endoscopy
[Insight AI Fellowship Project] This project demonstrates computer vision technology that can help doctors catch more gastrointestinal problems in patients in order to intervene and, for instance, prevent lethal colon cancers from developing.  I developed this project in consultation with the company Docbot over the course of five weeks while in the Insight Data Science AI Fellows program.

This repository contains python code that helps to partially automate analysis of gastrointestinal endoscopy images.  Approximately 140,000 private images were provided by Docbot via "pill-cams" (capsule endoscopy) for this project, only about 30 of which are publicly available in this repo.  The software here uses machine learning (convolutional neural networks for computer vision image classification) to classify the images.  It classifies images as being "normal" or "abnormal" (that is, containing a gastrointestinal abnormality such as blood, a polyp, or a lesion).  This flags a subset of the endoscopy images for a gastrointestinal doctor to examine further, reducing the amount of images to view and the time it takes to process the images, and helping increase the detection of abnormalities and intervention in cases that may lead to cancer and other problems.


## Contents of this Repo
|  folder                      | content |
| ---------------------------- | ------- |
| `ai\_vision\_for\_endoscopy` | the source code of the project |
| `explain`                    | material that explains the project and investigates the input and output |
| `input-data`                 | the endoscopy images (or the few that have been made public) |
| `output`                     | output files from the code |


## Running this Code
See the file `ai_vision_for_endoscopy/build/recommended_machine.txt` for a recommended machine and pre-configured environment to run the project's code.  With an appropriate machine and pre-configured environment, take the following steps in the shell terminal:

1. Clone this repo into your local machine and enter the repo:
    `git clone https://github.com/oneforawe/ai-vision-for-endoscopy`
    `cd ai-vision-for-endoscopy`

2. Create and activate a conda environment `analyzer_env` (to ensure the appropriate packages and versions are present):
    `conda env create -f environment.yml`
    `conda activate analyzer_env`

3. Test that you can run inference on the limited image data in the `input-data` folder.
    `cd ai_vision_for_endoscopy`
    `python analyzer_activator.py`

Check that the results (in the folder `output/infer/MNv2a/2-processed/by_abnorm/data_B/Round_03`) are similar to previously calculated results in the `Round_01` and `Round_02` folders.  In particular, examine the files in the `evaluations/figures/` folder and the `results/output_scores.csv` file.


## Further Context

### Problem
Colorectal cancer is the third most common cancer worldwide [1] and the second leading cause of cancer death in the US [2].  However, it's estimated that greater than 50% of these cancers are preventable with good quality colonoscopies [3].  In particular, colonoscopies can scale to more people with usage of pill-cams -- capsules with tiny cameras that patients swallow to have their whole gastrointestinal tract photographed, automating a whole gastrointestinal endoscopy.  To fully utilize this technology and the very large number of images generated, we need to at least partially automate the expert examination of the images to discover abnormalities that can be medically intervened upon, in some cases preventing death from cancers such as colon cancer.  Automated colonoscopy image analysis can significantly improve doctor's abnormality-detection rates and reduce their image viewing and analysis time by 80% or more.

### Solution
With computer vision algorithms, in particular those of convolutional neural networks, expert examination of gastrointestinal endoscopy images can be partially or fully automated.  Automated image analysis can significantly improve doctor's abnormality-detection rates and reduce their image viewing and analysis time by 80% or more.  The goal with this project was to build an AI endoscopy image analyzer able to classify images in a variety of ways, in particular "normal" and "abnormal", flagging the images with abnormalities for further examination by a gastrointestinal doctor.  More generally, this project provides a framework for iterating on algorithmic models to find better-performing models, with higher sensitivity and specificity.  This project was proposed by the company Docbot, and consulting with Docbot, I use their data and contextual understanding to inspire this project and make it a reality.

### References
[1] "Colorectal cancer was the third most common cancer [worldwide in 2018]..."
World Cancer Research Fund, American Institute for Cancer Research
https://www.wcrf.org/dietandcancer/cancer-trends/worldwide-cancer-data
(Accessed Feb 2019)
Source: Bray F, Ferlay J, Soerjomataram I, Siegel RL, Torre LA, Jemal A. Global Cancer Statistics 2018: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA Cancer J Clin, in press. The online GLOBOCAN 2018 database is accessible at http://gco.iarc.fr/, as part of IARC's Global Cancer Observatory.

[2] "In the United States, [as of 2019,] colorectal cancer is ... the second most common cause of cancer deaths when men and women are combined."
American Cancer Society
https://www.cancer.org/cancer/colon-rectal-cancer/about/key-statistics.html
(Accessed Feb 2019)

[3] "Colorectal cancer: [As of May 2018] It's estimated that more than half of all cases could be prevented by regular colonoscopy screening!"
American Cancer Society
https://www.cancer.org/content/dam/cancer-org/cancer-control/en/presentations/colorectal-cancer-presentation.pdf
(Accessed Feb 2019)

Note:

[4] "About 90 percent of colorectal cancers and deaths are thought to be preventable.  In addition to regular colorectal cancer screenings, exercise and maintaining a healthy weight can reduce your risk of colorectal cancer."
UCSF Health
https://www.ucsfhealth.org/education/colorectal_cancer_prevention_and_screening/
(Accessed Feb 2019)

