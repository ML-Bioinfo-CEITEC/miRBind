# to run the TargetNet evaluation you need to install it first
# run: pip install git+https://github.com/katarinagresova/TargetNet.git
# you also need to donwload a pretrained TargetNet model into Models
# wget https://github.com/katarinagresova/TargetNet/blob/master/pretrained_models/TargetNet.pt?raw=true -O Models/TargetNet.pt
# 
# Problem of this method is that is was trained on miRNA-mRNA pairs where the 1. nucleotide of miRNA was positioned at the 6. nucleotide of mRNA
# They are doing sequence alignment of 1.-10. nucleotide of miRNA and 6.-16. nucleotide of mRNA
# However, in our data, miRNA is more centered in the mRNA
#
# Important changes to the original implementation:
# 1) mRNA sequences were shortened to 40 nucleotides by removing the first 5 nucleotides and the last 5 nucleotides
# 2) mRNA sequences were not reversed
# 3) miRNA-mRNA seed alignment was done from position 10 to 20 on mRNA (miRNA is centered in mRNA in our data)

import sys

import torch
import pandas as pd
import numpy as np
from Bio import pairwise2
from targetnet.model.model_utils import get_model
from targetnet.train import Trainer
from targetnet.utils import set_seeds, set_output
from targetnet.data import reverse, extended_seed_alignment, encode_RNA

# This class is changed compared to the original TargetNet implementation
# We don't use ids
# But we keep the original sequences so we can report them later together with the predictions (and it will not be mixed up even if we shuffle the dataset)
class miRNA_CTS_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for miRNA-CTS pair data """
    def __init__(self, X, labels, miRNAs, mRNAs):
        self.X = X
        self.labels = labels
        self.miRNAs = miRNAs
        self.mRNAs = mRNAs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.X[i], self.labels[i], self.miRNAs[i], self.mRNAs[i]

score_matrix = {}  # Allow wobble
for c1 in 'ACGU':
    for c2 in 'ACGU':
        if (c1, c2) in [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')]:
            score_matrix[(c1, c2)] = 1
        elif (c1, c2) in [('U', 'G'), ('G', 'U')]:
            score_matrix[(c1, c2)] = 1
        else:
            score_matrix[(c1, c2)] = 0

def extended_seed_alignment(mi_seq, cts_r_seq):
    """ extended seed alignment """
    alignment = pairwise2.align.globaldx(mi_seq[:10], cts_r_seq[10:20], score_matrix, one_alignment_only=True)[0]
    mi_esa = alignment[0]
    cts_r_esa = alignment[1]
    esa_score = alignment[2]
    return mi_esa, cts_r_esa, esa_score

def encode_RNA(mirna_seq, mirna_esa, cts_rev_seq, cts_rev_esa):
    """ one-hot encoder for RNA sequences with extended seed alignments """
    chars = {"A":0, "C":1, "G":2, "U":3, "-":4}
    x = np.zeros((len(chars) * 2, 50), dtype=np.float32)
    # ENCODING miRNA part
    # 1) positions [0,4] are zeros
    # 2) positions [5,5+len(mirna_esa)] are encoded acccording to mirna_esa
    for i in range(len(mirna_esa)):
        x[chars[mirna_esa[i]], 5 + i] = 1
    # 3) positions [5+len(mirna_esa), 5+len(mirna_seq)] are encoded according to mirna_seq
    for i in range(10, len(mirna_seq)):
        x[chars[mirna_seq[i]], 5 + len(mirna_esa) + i - 10 ] = 1
    # 4) positions till 50 are zeros

    #ENCODING mRNA part
    # 1) positions [0,9] are encoded according to cts_rev_esa
    for i in range(10):
        x[chars[cts_rev_seq[i]] + len(chars), i] = 1
    # 2) positions [10,10+len(cts_rev_esa)] are encoded according to cts_rev_esa
    for i in range(len(cts_rev_esa)):
        x[chars[cts_rev_esa[i]] + len(chars), i + 10] = 1
    # 3) positions till 50 are encoded according to cts_rev_seq
    for i in range(20, len(cts_rev_seq)):
        x[chars[cts_rev_seq[i]] + len(chars), i + len(cts_rev_esa) + 10 - 20] = 1

    return x


def get_dataset_from_file(file_path):
    """ load miRNA-CTS dataset from file """
    FILE = open(file_path, "r")
    lines = FILE.readlines()
    FILE.close()

    X, labels, miRNAs, mRNAs = [], [], [], []
    for l, line in enumerate(lines[1:]):
        tokens = line.strip().split("\t")
        mirna_seq, mrna_seq = tokens[:2]

        miRNAs.append(mirna_seq)
        mRNAs.append(mrna_seq)

        label = float(tokens[2]) if len(tokens) > 2 else 0

        mirna_seq = mirna_seq.upper().replace("T", "U")
        mrna_seq = mrna_seq[5:45]
        mrna_seq = mrna_seq.upper().replace("T", "U")
        #mrna_rev_seq = reverse(mrna_seq)
        mrna_rev_seq = mrna_seq

        mirna_esa, cts_rev_esa, esa_score = extended_seed_alignment(mirna_seq, mrna_rev_seq)
        X.append(torch.from_numpy(encode_RNA(mirna_seq, mirna_esa, mrna_rev_seq, cts_rev_esa)))
        labels.append(torch.from_numpy(np.array(label)).unsqueeze(0))
        

    dataset = miRNA_CTS_dataset(X, labels, miRNAs, mRNAs)

    return dataset

class ModelConfig():
    def __init__(self, cfg, idx="model_config"):
        """ model configurations """
        self.idx = idx
        self.type = None
        self.num_channels = None
        self.num_blocks = None
        self.stem_kernel_size = None
        self.block_kernel_size = None
        self.pool_size = None

        for key, value in cfg.items():
            if key == "skip_connection":                self.skip_connection = value
            elif key == "num_channels":                 self.num_channels = value
            elif key == "num_blocks":                   self.num_blocks = value
            elif key == "stem_kernel_size":             self.stem_kernel_size = value
            elif key == "block_kernel_size":            self.block_kernel_size = value
            elif key == "pool_size":                    self.pool_size = value
            else: sys.exit("# ERROR: invalid key [%s] in model-config file" % key)

def evaluate(model, dataloader):

    model.eval()

    all_miRNAs = []
    all_mRNAs = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for X, y, miRNA, mRNA in dataloader:

            output = model(X)

            pred = torch.sigmoid(output)

            all_miRNAs.extend(miRNA)
            all_mRNAs.extend(mRNA)
            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())


    return all_miRNAs, all_mRNAs, all_labels, all_predictions


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(2020)
    
    model_cfg = ModelConfig({
        'skip_connection': True,
        'num_channels': [16, 16, 32],
        'num_blocks': [2, 1, 1],
        'stem_kernel_size': 5,
        'block_kernel_size': 3,
        'pool_size': 3
    })
    output, save_prefix = set_output({'output_path': 'TargetNet-evaluation/'}, "evaluate_model_log")

    model, params = get_model(model_cfg, with_esa=True)

    trainer = Trainer(model)
    trainer.load_model("../Models/TargetNet.pt", output)
    trainer.set_device(device)

    for dataset_ratio in ['1', '10', '100']:
        
        dset = get_dataset_from_file("../Datasets/test_set_1_" + dataset_ratio + "_CLASH2013_paper.tsv")
        dataloader = torch.utils.data.DataLoader(dset, 64, shuffle=False, pin_memory=True, num_workers=2)

        miRNAs, mRNAs, labels, predictions = evaluate(trainer.model, dataloader)
        df = pd.DataFrame({"miRNA": miRNAs, "mRNA": mRNAs, "label": labels, "prediction": predictions})
        df['label'] = df['label'].apply(lambda x: int(x[0]))
        df['prediction'] = df['prediction'].apply(lambda x: x[0])
        df.to_csv('targetnet_metrics/targetnet_score_1_' + str(dataset_ratio) + '.tsv' ,index=False, sep='\t')