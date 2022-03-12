#!/usr/bin/env python

import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification
import torch
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
from pathlib import Path


def preprocess_datasets(dset, tokenizer, kmer_len, stride):
    '''
    Currently 
    AABBC with stride = 2 and kmer = 2 produces
    ['AA','BB'] and cuts off C 
    TODO is this desired? If not, set for cycle limit to len(text) only
    '''

    offset = kmer_len

    #This is where the max_length is computed
    max_seq_len = max([len(dset[i][0]) for i in range(len(dset))])
    max_seq_len = max_seq_len if max_seq_len < 512 else 512 

    if(max_seq_len >=512):
        print('WARNING: some sequences are longer than 512, these are being trimmed')

    def coll(data):
        encoded_dset = [(label, tokenizer([text[i:i+offset] 
                        for i in range(0, len(text)-offset+1, stride)], max_length=max_seq_len, padding="max_length", is_split_into_words=True, truncation=True, verbose=True).input_ids)
                        for text, label in data]
        encoded_samples = [{"input_ids": torch.tensor(ids), "attention_mask": torch.tensor([1]*len(ids)), "labels": torch.tensor(label)} 
                    for label, ids in encoded_dset]

        return encoded_samples

    encoded_samples = coll(dset)

    return encoded_samples


def get_predictions_and_labels(encoded_samples_test, model):
    test_loader = DataLoader(
                encoded_samples_test, 
                sampler = SequentialSampler(encoded_samples_test), 
                batch_size = 4 #TODO increase with your CPU
            )

    predictions = []
    # for sample in tqdm(test_loader, total=len(test_dset)/32):

    for sample in tqdm(test_loader, total=len(test_loader), desc='Predicting', position=1): 

        outputs = model.to("cpu")(**sample)
        # outputs = model(**sample) #TODO make eval on GPU

        #preds = outputs.logits.argmax(-1).tolist()
        preds = np.array(outputs.logits.tolist())
        predictions.extend(preds)

    #labels = pd.read_csv(test_data, sep='\t', usecols=['label']).to_numpy()

    return predictions#, labels


if __name__ == "__main__":

    if not Path('../Models/dnabert_for_clash_1_1').exists():
        print('dnabert model not found. You might need to download it from Drive.')
        exit()

    for dataset_ratio in ['1', '10', '100']:

        model = AutoModelForSequenceClassification.from_pretrained("../Models/dnabert_for_clash_1_1/")
        test_dset_iter = pd.read_csv("../Datasets/test_set_1_" + dataset_ratio + "_CLASH2013_paper.tsv", sep='\t', iterator=True, chunksize=5000)
        
        header = True
        for chunk in tqdm(test_dset_iter, desc='Chunks', position=0):
            test_dset = pd.DataFrame(columns=['miRNA', 'gene', 'label'])
            test_dset = pd.concat([test_dset, chunk], ignore_index=True)

            test_dset['seq'] = test_dset.apply(lambda x: x['miRNA'] + 'NNNN' + x['gene'], axis=1)

            kmer_len = 6
            stride = 1
            tokenizer = AutoTokenizer.from_pretrained(f"armheb/DNA_bert_{kmer_len}")
            processed_dset = preprocess_datasets(test_dset[['seq', 'label']].to_numpy(), tokenizer, kmer_len, stride)

            predictions = get_predictions_and_labels(processed_dset, model)
            probs = softmax(predictions, axis=1)[:, 1]
            test_dset['dnabert'] = probs

            test_dset[['miRNA', 'gene', 'label', 'dnabert']].to_csv('dnabert_metrics/dnabert_score_1_' + str(dataset_ratio) + '.tsv', index=False, mode='a', header=header, sep='\t')
            header = False