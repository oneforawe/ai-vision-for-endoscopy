#!/usr/bin/env python3
# filename: timer.py

import os
import datetime


def record_duration(interval, description, filepath):
    days = interval.days
    secs = interval.seconds
    hrs  = secs//3600
    mins = (secs-hrs*3600)//60
    secs = secs - hrs*3600 - mins*60
    file = open(filepath, 'w')
    file.write(f'{description}\n = ' +
               f'{days} days, {hrs} hours, ' +
               f'{mins} minutes, {secs} seconds, ' +
               f'{interval.microseconds} microseconds.\n')
    file.close()


