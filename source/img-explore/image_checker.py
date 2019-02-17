#!/usr/bin/env python3
# filename: image_checker.py
# version:  1

import os
import glob
import cv2
from tqdm import tqdm


###########################
# Characterize Image Data #
###########################


def main():

    # path to image data
    path = '../data/1-pre-processed/A'

    filePathList = []
    mpgPathList = []
    for dirPath, subDirs, files in os.walk(path):
        for file in files:
            if file.endswith( (".jpg", ".jpeg") ):
                filePathList.append( os.path.join(dirPath, file) )
            if file.endswith( (".mpg", ".mpeg") ):
                mpgPathList.append( os.path.join(dirPath, file) )
    print('')
    print(f'Number of (pre-processed) .jpg & .jpeg files: {len(filePathList)}')
    print(f'Number of (pre-processed) .mpg & .mpeg files: {len(mpgPathList)}')

    #...
    shapeList = []
    print('')
    print('Inspecting each file to find how many unique shapes (height, ' + \
          'width, channels) there are.  May take several minutes (possibly ' + \
          '~8 min or more).')
    for filePath in tqdm(filePathList):
        img = cv2.imread(filePath)
        shapeList.append(img.shape)
    uniqueShapes = set(shapeList)
    print('')
    print(f'For images (jpg, jpeg files):')
    print(f'Number of unique image shapes: {len(uniqueShapes)}')
    print(f'The unique image shapes:       {uniqueShapes}')
    print('')


if __name__ == '__main__':
    main()

