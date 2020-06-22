'''
@author: Prafull SHARMA
@Data Set: Kaggle Toxc comment dataset
'''

from __future__ import print_function, division
from builtins import range
# for updating in future
# using pip install -U future
# ----------- Utils ---------------------------------
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ------------ NN libs ------------------------------
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPool1D
from keras.layers import Conv1D, MaxPool1D, Embedding
# ----------------------------------------------------
from configurations import *

# Embedding path...
emb_path = os.path.expanduser("~") + f'/EMBEDDINGS/glove.6B.{EMBEDDIN_DIM}d.txt'
# words pointing to vectors dict
with open(emb_path, 'r') as fil:
    word2vec = {
                    line.split()[0]: np.asarray(line.split()[1:])
                    for line in fil
                }
    print(f"FOUND: {len(word2vec)} word vectors")



