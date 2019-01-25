#!/usr/bin/env bash

"""
printf "Number of .jpg  files: " >> image-file-count.txt
find . -type f -name "*.jpg" | wc -l >> image-file-count.txt
printf "\n" >> image-file-count.txt

printf "Number of .jpeg files: " >> image-file-count.txt
find . -type f -name "*.jpeg" | wc -l >> image-file-count.txt
printf "\n" >> image-file-count.txt

printf "Number of .mpg  files: " >> image-file-count.txt
find . -type f -name "*.mpg" | wc -l >> image-file-count.txt
printf "\n" >> image-file-count.txt
"""

printf "Number of .jpg  files: "
find . -type f -name "*.jpg" | wc -l
printf "Number of .jpeg files: "
find . -type f -name "*.jpeg" | wc -l
printf "Number of .mpg  files: "
find . -type f -name "*.mpg" | wc -l

