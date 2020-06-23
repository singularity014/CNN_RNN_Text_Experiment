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
from keras.models import Model
from keras.layers import Conv1D, MaxPool1D, Embedding
# ----------------------------------------------------
from configurations import *
from sklearn.metrics import roc_auc_score
# ----------------------------------------------------


# Embedding path...
emb_path = os.path.expanduser("~") + f'/EMBEDDINGS/glove.6B.{EMBEDDIN_DIM}d.txt'
# words pointing to vectors dict
with open(emb_path, 'r') as fil:
    word2vec = {
                    line.split()[0]: np.asarray(line.split()[1:])
                    for line in fil
                }
    print(f"FOUND: {len(word2vec)} word vectors")

# prepare Text samples and Labels
print('Loading comments ....')
train = pd.read_csv('data/kaggle_toxic_comment_challenge/train.csv')
sentences = train['comment_text'].fillna("DUMMY_VALUE").values
possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = train[possible_labels].values

# analysing min, max of sequence length
seq_length = np.array([len(s.split()) for s in sentences])
# print(np.min(seq_length))
# print(np.max(seq_length))
# print(int(np.mean(seq_length)))

# Tokenization and index creation
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# word2index  : word ---> integer mapping
word2idx = tokenizer.word_index
print(f"FOUND: {len(word2idx)} unique tokens in data...")

# Paadding to make all the vectors
# in the sequences of same lentgh  [N x T]
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(f"Shape of data tensor: {data.shape}")

# EMBEDDING MATRIX prepare
print('Filling pre-trained embedding matrix....')
num_words = min(MAX_VOCAB_SIZE, len(word2idx)+1)
# shape = V x D
embedding_matrix = np.zeros((num_words, EMBEDDIN_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word, None)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
print(embedding_matrix.shape)


# ----------------- MODEL DEVELOPMENT -------------------------

# 1) EMBEDDING LAYER .......
embedding_layer = Embedding(
    num_words,
    EMBEDDIN_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False
)

# 2) MODEL LAYERS...........................................
# Creating a 1-D ConvNet with Global Max POoling
# Since the input is size N X T ,,,so we pass T
# which is MAX_SEQUENCE_LENGTH
input_ = Input(shape=(MAX_SEQUENCE_LENGTH, ))
layer = embedding_layer(input_)
layer = Conv1D(128, 3, activation='relu')(layer)
layer = MaxPool1D(3)(layer)
layer = Conv1D(128, 3, activation='relu')(layer)
layer = MaxPool1D(3)(layer)
layer = Conv1D(128, 3, activation='relu')(layer)
layer = GlobalMaxPool1D()(layer)
layer = Dense(128, activation='relu')(layer)
output = Dense(len(possible_labels), activation='sigmoid')(layer)

# ----------------- MODEL COMPILE -------------------------
model = Model(input_, output)
model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# ----------------- MODEL TRAINING --------------------------
train = model.fit(
    data,
    targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCH,
    validation_split=VALIDATION_SPLIT
)

# ---------------- METRICS CHECK ----------------------------
# losses
plt.plot(train.history['loss'], label='loss')
plt.plot(train.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(train.history['acc'], label='acc')
plt.plot(train.history['val_acc'], label='val_loss')
plt.legend()
plt.show()

# --------- AUC check -------------------------------------
predictions = model.predict(data)
# Note: Data has been split automatically into test, thanks to keras feature!
# This can be seen in model.fit 'VALIDATION SPLIT'
aucs= []
for i in range(6):
    auc = roc_auc_score(targets[:, i], predictions[:, i])
    aucs.append(auc)
    
print()
print(f"AREA UNDER CURVE : {np.mean(aucs)}")
