import os
import sys


def get_root():
    return os.path.dirname(os.path.abspath(__file__))

def add_par_dir():
    sys.path.append(os.path.join(get_root(), ".."))
