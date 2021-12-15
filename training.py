#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
from tensorflow.keras.layers import (
                                BatchNormalization, LeakyReLU,
                                Input, Dense, Conv2D,
                                MaxPooling2D, Flatten, Dropout)
from tensorflow.keras.optimizers import Adam


def one_hot_encoding(df, tensor_dim=(50, 20, 1)):
    """
    fun transform input database to numpy array.
    
    parameters:
    df = Pandas df with col names "gene", "label", "miRNA"
    tensor_dim= 2d matrix shape
    
    output:
    2d dot matrix, labels as np array
    """
    df.reset_index(inplace=True, drop=True)

    # alphabet for watson-crick interactions.
    alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1.}

    # labels to one hot encoding
    label = df["label"].to_numpy()

    # create empty main 2d matrix array
    N = df.shape[0]  # number of samples in df
    shape_matrix_2d = (N, *tensor_dim)  # 2d matrix shape
    # initialize dot matrix with zeros
    ohe_matrix_2d = np.zeros(shape_matrix_2d, dtype="float32")

    # compile matrix with watson-crick interactions.
    for index, row in df.iterrows():
        for bind_index, bind_nt in enumerate(row.gene.upper()):

            for mirna_index, mirna_nt in enumerate(row.miRNA.upper()):

                base_pairs = bind_nt + mirna_nt
                ohe_matrix_2d[index, bind_index, mirna_index, 0] = alphabet.get(base_pairs, 0)

    return ohe_matrix_2d, label


def make_architecture():
    """
    build model architecture

    return a model object
    """
    main_input = Input(shape=(50, 20, 1),
                       dtype='float32', name='main_input'
                       )

    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        data_format="channels_last",
        name="conv_1")(main_input)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Max_1')(x)
    x = Dropout(rate=0.25)(x)

    x = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        data_format="channels_last",
        name="conv_2")(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Max_2')(x)
    x = Dropout(rate=0.25)(x)

    x = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        data_format="channels_last",
        name="conv_3")(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Max_3')(x)
    x = Dropout(rate=0.25)(x)

    conv_flat = Flatten(name='2d_matrix')(x)

    x = Dense(128)(conv_flat)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)

    x = Dense(64)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)

    x = Dense(32)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)

    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    m = K.Model(inputs=[main_input], outputs=[main_output], name='arch_00')

    return m


def compile_model():
    K.backend.clear_session()
    m = make_architecture()

    opt = Adam(
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam")

    m.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
        )
    return m


def plot_history(history):
    """
    plot history of the training of the model,
    accuracy and loss of the training and validation set
    """

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 6), dpi=80)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    plt.show()


training_ratio = 1
train_df = pd.read_csv("Datasets/train_set_1_" + str(training_ratio) + "_CLASH2013_paper.tsv", sep='\t')
# set random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
train_df = train_df.sample(frac=1, random_state=RANDOM_STATE)
print(train_df.head())
ohe_data = one_hot_encoding(train_df)
train_ohe, labels = ohe_data
print("Number of training samples: ", train_df.shape[0])

model = compile_model()
model.summary()

model_history = model.fit(
    train_ohe, labels,
    validation_split=0.05, epochs=10,
    batch_size=32,
    class_weight={0: 1, 1: training_ratio}
    )

plot_history(model_history)

model.save("model_1_" + str(training_ratio) + ".h5")
