#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from tensorflow import keras as K
import tensorflow_addons as tfa

def load_cofold(name):
    f = open(name, 'r')
    scores = []
    while True:
        _ = f.readline()
        _ = f.readline()
        line3 = f.readline()
        if not line3:
            break
        ll = line3.split()
        score = -float(ll[-1].replace("(", "")[:-2])
        scores.append(score)
    minimum = min(scores)
    maximum = max(scores)
    norm = [(val - minimum) / (maximum - minimum) for val in scores]
    return norm, scores


def load_RNAhybrid(name):
    df = pd.read_csv(name, sep='\t', header=None, names=['miRNA', 'mRNA', 'score', 'label'])
    maximum = (-1)*df['score'].min()
    minimum = (-1)*df['score'].max()
    df['score'] = df['score'].apply(lambda x: ((-1)*x - minimum)/(maximum - minimum))
    return np.array(df['label']), np.array(df['score'])


def load_rna22(name, pos_count, neg_count):
    """_summary_

    Args:
        name (str): path to data file
        pos_count (int): count of positive examples in original test file
        neg_count (int): count of negative examples in original test file

    Returns:
        tuple(np.array, np.array): array of labels and array of normalized score
    """

    rna22 = pd.read_csv(name, header=None, usecols=[10,11], names=['score', 'label'], sep='\t')
    # they use p-value so we need to invert in to use it as score
    rna22['score'] = rna22['score'].apply(lambda x: 1 - x)

    # normalizing to the whole 0-1 interval
    minimum = min(rna22['score'])
    maximum = max(rna22['score'])
    rna22['norm'] = rna22['score'].apply(lambda x: (x - minimum) / (maximum - minimum))

    # we don't have results for all sequences - adding score 0 for missing ones
    pos = pos_count - rna22['label'].value_counts()[1]
    neg = neg_count - rna22['label'].value_counts()[0]
    added_score = []
    for i in range(pos):
        added_score.append([0, 1, 0])
    for i in range(neg):
        added_score.append([0, 0, 0])

    rna22 = pd.concat([rna22, pd.DataFrame(added_score, columns=['score', 'label', 'norm'])], ignore_index=True)

    return np.array(rna22['label']), np.array(rna22['norm'])


def load_dnabert(dataset_ratio):
    df = pd.read_csv('dnabert_metrics/dnabert_score_1_' + dataset_ratio + '.tsv', sep='\t')
    return np.array(df['label']), np.array(df['dnabert'])


def load_resnet(dataset_ratio):
    df = pd.read_csv('resnet_preds/resnet_preds_' + dataset_ratio + '.csv')
    return np.array(df['labels']), np.array(df['model_pred'])


def one_hot_encoding(df, tensor_dim=(50, 20, 1)):
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


def seed_match(miRNA, gene):
    alphabet = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    seed = miRNA[1:7]
    seed = ''.join([alphabet[s] for s in seed][::-1])
    return seed in gene


def seed_pr(data):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for index, row in data.iterrows():
        if seed_match(row['miRNA'].upper(), row['gene'].upper()):
            if row['label'] == 1:
                TP += 1
            else:
                FP += 1
        elif row['label'] == 1:
            FN += 1
        else:
            TN += 1
    return TP / (TP + FP), TP / (TP + FN)


# PARAMETERTS - you might want to change these
models_params = {
    'CNN1': {
        'color': '#a9d9f0',
        'label': 'CNN 1:1'
    },
    'CNN10': {
        'color': '#5b8ea5',
        'label': 'CNN 1:10'
    },
    'CNN100': {
        'color': '#2ab9cc',
        'label': 'CNN 1:100'
    },
    'Cofold': {
        'color': '#702601',
        'label': 'Cofold'
    },
    'miRBind': {
        'color': 'navy',
        'label': 'miRBind'
    },
    'dnabert': {
        'color': '#ff4181',
        'label': 'DNABERT'
    },
    'rna22': {
        'color': '#ffcc00',
        'label': 'RNA22'
    },
    'seed': {
        'color': '#ff7628',
        'label': 'Seed'
    },
    'RNAhybrid': {
        'color': '#7917a6',
        'label': 'RNAhybrid'
    }
}
models = ['miRBind', 'dnabert', 'CNN1', 'rna22', 'Cofold', 'RNAhybrid', 'seed']
dataset_ratio = '1'
fig_name = 'PR_test_set_1_' + dataset_ratio + '.png'
# END of parameters


df = pd.read_csv("../Datasets/test_set_1_" + dataset_ratio + "_CLASH2013_paper.tsv", sep='\t')
ohe_data = one_hot_encoding(df)
seq_ohe, labels = ohe_data
print("Number of samples: ", df.shape[0])

plt.figure(figsize=(4, 4), dpi=250)

if 'CNN1' in models:
    model_1 = K.models.load_model("../Models/CNN_model_1_1_optimized.h5")
    model_1_predictions = model_1.predict(seq_ohe)
    precision, recall, _ = precision_recall_curve(labels, model_1_predictions)
    print("CNN 1:1 auc", metrics.auc(recall, precision))
    plt.plot(recall, precision, label=models_params['CNN1']['label'], marker=',', color=models_params['CNN1']['color'])

if 'CNN10' in models:
    model_10 = K.models.load_model("../Models/CNN_model_1_10_optimized.h5")
    model_10_predictions = model_10.predict(seq_ohe)
    precision, recall, _ = precision_recall_curve(labels, model_10_predictions)
    print("CNN 1:10 auc", metrics.auc(recall, precision))
    plt.plot(recall, precision, label=models_params['CNN10']['label'], marker=',', color=models_params['CNN10']['color'])

if 'CNN100' in models:
    model_100 = K.models.load_model("../Models/CNN_model_1_100_optimized.h5")
    model_100_predictions = model_100.predict(seq_ohe)
    precision, recall, _ = precision_recall_curve(labels, model_100_predictions)
    print("CNN 1:100 auc", metrics.auc(recall, precision))
    plt.plot(recall, precision, label=models_params['CNN100']['label'], marker=',', color=models_params['CNN100']['color'])

if 'Cofold' in models:
    cofold, cofold_orig = load_cofold("../Datasets/test_set_1_" + dataset_ratio + "_CLASH2013_paper_cofold.fasta")
    precision, recall, _ = precision_recall_curve(labels, np.array(cofold))
    print("Cofold auc ", metrics.auc(recall, precision))
    plt.plot(recall, precision, label=models_params['Cofold']['label'], marker=',', color=models_params['Cofold']['color'])

if 'rna22' in models:
    rna22_labels, rna22_probs = load_rna22("../Datasets/test_set_1_" + dataset_ratio + "_CLASH2013_paper_rna22.txt", 2000, int(dataset_ratio)*2000)
    precision, recall, _ = precision_recall_curve(rna22_labels, rna22_probs)
    print("RNA22 auc ", metrics.auc(recall, precision))
    plt.plot(recall, precision, label=models_params['rna22']['label'], marker=',', color=models_params['rna22']['color'])

if 'RNAhybrid' in models:
    RNA_hybrid_label, RNA_hybrid_score = load_RNAhybrid("../Datasets/test_set_1_" + dataset_ratio + "_CLASH2013_paper_RNAhybrid_scores.txt")
    precision, recall, _ = precision_recall_curve(RNA_hybrid_label, RNA_hybrid_score)
    print("RNAhybrid auc ", metrics.auc(recall, precision))
    plt.plot(recall, precision, label=models_params['RNAhybrid']['label'], marker=',', color=models_params['RNAhybrid']['color'])

if 'dnabert' in models:
    dnabert_labels, dnabert_probs = load_dnabert(dataset_ratio)
    precision, recall, _ = precision_recall_curve(dnabert_labels, dnabert_probs)
    print("DNABERT auc ", metrics.auc(recall, precision))
    plt.plot(recall, precision, label=models_params['dnabert']['label'], marker=',', color=models_params['dnabert']['color'])

if 'miRBind' in models:
    resnet_labels, resnet_probs = load_resnet(dataset_ratio)
    precision, recall, _ = precision_recall_curve(resnet_labels, resnet_probs)
    print("miRBind auc ", metrics.auc(recall, precision))
    plt.plot(recall, precision, label=models_params['miRBind']['label'], marker=',', color=models_params['miRBind']['color'])

if 'seed' in models:
    prec, sens = seed_pr(df)
    plt.plot(sens, prec, label=models_params['seed']['label'], marker='.', color=models_params['seed']['color'])
    print("Seed sens, prec: ", sens, prec)

plt.legend(loc='best')
plt.axis([0, 1, 0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curve on 1:' + dataset_ratio + ' test set')
plt.savefig(fig_name)
