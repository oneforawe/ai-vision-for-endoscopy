#!/usr/bin/env python3
# filename: img_culler.py

import os
from sklearn.utils import shuffle

# Goal:
# For folder E, get down from 138,062 files to 10,000.
# Calculated on a spreadsheet how many to delete from each subfolder,
# while keeping some balance in Abnormal/Normal and each region.
# So:
# Given a folder with many, many image files (~30k),
# delete some number of files (num_to_delete),
# but delete them in a random fashion.

# Strategy:
# 1) get list of all files in the folder of interest
# 2) randomize list
# 3) delete first num_to_delete files


def main():

    # Location to cull files:
    src_path_root = '../../input-data/1-pre-processed/E/'
    src_groups = ['Abnormal', 'Normal']
    src_regions = ['Colon', 'Esophagus', 'Small bowel', 'Stomach']
    original_numbers = [ [20479, 243, 29949, 1504], [31798, 1380, 34028, 18681] ]
    numbers_to_delete = [ [19082, 0, 28553, 108], [30402, 0, 32632, 17285] ]
    # (Don't want to cull esophagus files since there aren't many of those.)
    #resulting_numbers = [ [1397, 243, 1396, 1396], [1396, 1380, 1396, 1396] ]
    #                  = [       [tot 4432],            [tot 5568]           ]
    #                  = [                   tot 10000                       ]

    # For all folders,
    for i, group in enumerate(src_groups):
        for j, region in enumerate(src_regions):
            # given a folder,
            src_path = src_path_root + group + '/' + region + '/'
            # list files.
            filePathList = []
            for dirPath, subDirs, files in os.walk(src_path):
                for file in files:
                    if file.endswith(".jpg"):
                        filePathList.append( os.path.join(dirPath, file) )
            # If these numbers are correct (and culling hasn't happened yet),
            if original_numbers[i][j] == len(filePathList):
                # procede to cull files.
                # Randomize list.
                filePathList = shuffle(filePathList)
                num_to_delete = numbers_to_delete[i][j]
                # Delete the first num_to_delete.
                for filePath in filePathList[:num_to_delete]:
                    os.remove(filePath)


if __name__ == '__main__':
    main()


# Notes
# -----
# Needed to move some Colon images to Small bowel
# (since they were initially misplaced).
# Initial / Final comparisons:                     here        to here
# original_numbers = [ [20479, 243, 29949, 1504], [31847, 1380, 33979, 18681] ]
# original_numbers = [ [20479, 243, 29949, 1504], [31789, 1380, 34028, 18681] ]
#                                                    moved 58 files
# numbers_to_delete = [ [19082, 0, 28553, 108], [30451, 0, 32583, 17285] ]
# numbers_to_delete = [ [19082, 0, 28553, 108], [30402, 0, 32632, 17285] ]

