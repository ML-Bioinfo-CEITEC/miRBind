#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
from tensorflow import keras as k


def parse_input():
    """
    function for parsing input parameters
    :return: dictionary of parameters
    """
    parser = argparse.ArgumentParser(description='miRBind: a method for prediction of potential miRNA:target site '
                                                 'binding')
    parser.add_argument('--input', default="example.tsv", metavar='<input_tsv_filename>')
    parser.add_argument('--output', default="example_scores", metavar='<output_filename_prefix>')
    parser.add_argument('--model', default="Models/model_1_1.h5", metavar='<model_name>')
    args = parser.parse_args()
    return vars(args)


def one_hot_encoding(df, tensor_dim=(50, 20, 1)):
    """
    fun encodes miRNAs and mRNAs in df into binding matrices
    :param df: dataframe containing 'gene' and 'miRNA' columns
    :param tensor_dim: output shape of the matrix
    :return: numpy array of predictions
    """
    # alphabet for watson-crick interactions.
    alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1., "AU": 1., "UA": 1.}
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

    return ohe_matrix_2d


def write_score(output_file, df, scores):
    """
    fun writes information about sequence and its score to the output_file
    :param output_file
    :param df: dataframe with miRNA:target pairs
    :param scores: numpy array, predicted scores
    """
    scores = scores.flatten()
    df["score"] = pd.Series(scores, index=df.index)
    df.to_csv(output_file + '.tsv', sep='\t', index=False)


def predict_probs(df, model, output):
    """
    fun predicts the probability of miRNA:target site binding in df file
    :param df: input dataframe with sequences containing 'gene' and 'miRNA' columns
    :param model: Keras model used for predicting
    :param output: output file to write probabilities to
    """
    miRNA_length = 20
    gene_length = 50

    orig_len = len(df)
    mask = (df["miRNA"].str.len() == miRNA_length) & (df["gene"].str.len() == gene_length)
    df = df[mask]
    processed_len = len(df)

    if orig_len != processed_len:
        print("Skipping " + str(orig_len - processed_len) + " pairs due to inappropriate length.")

    ohe = one_hot_encoding(df)
    prob = model.predict(ohe)
    write_score(output, df, prob)


def main():
    arguments = parse_input()

    output = arguments["output"]

    try:
        model = k.models.load_model(arguments["model"])
    except (IOError, ImportError):
        print()
        print("Can't load the model", arguments["model"])
        return

    print("===========================================")

    try:
        input_df = pd.read_csv(arguments["input"], names=['miRNA', 'gene'], sep='\t')
    except IOError as e:
        print()
        print("Can't load file", arguments["input"])
        print(e)
        return

    predict_probs(input_df, model, output)


main()
