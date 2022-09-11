import os
import sys


def find_parent_filepath():
    wrk_dir = os.getcwd()

    while wrk_dir[-26:] != 'speech-emotion-recognition':
        wrk_dir = os.path.dirname(wrk_dir)

    return wrk_dir
