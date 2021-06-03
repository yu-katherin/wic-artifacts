#!/usr/bin/env python3 -u

import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import copy
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModel
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, f1_score

from dataset import process_wic_data


def get_subidx(x, y):
    '''
    Used to get the start index of the subword sequence corresponding to the token of 
    interest.
    '''
    l1, l2 = len(x), len(y)
    for i in range(l1):
        if x[i:i+l2] == y:
            return i
    raise ValueError()

    import string, re


def process_hidden_states(df):
    hidden_states = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for i in range(df.shape[0]):
        if i % 100 == 0: 
            print(i)
        dterm_idx1, dterm_idx2  = df.loc[i, 'ii'].split('-')
        hs = np.array([])
        for j in range(2): 
            text_seq = df.loc[i, "sentence1"] if j == 0 else df.loc[i, "sentence2"]

            # run through encoder to extract hidden states
            encoded_input = tokenizer(text_seq, return_tensors='pt')
            output = model(
                input_ids=encoded_input['input_ids'].to(device),
                attention_mask= ncoded_input['attention_mask'].to(device),
                token_type_ids=encoded_input['token_type_ids'].to(device),
                output_hidden_states=True,
            )

            # get the actual target word ('carry' versus 'carries') from the text using given index
            text_seq_split = text_seq.split()
            dterm = text_seq_split[int(dterm_idx1)] if j == 0 else text_seq_split[int(dterm_idx2)]
            dterm_converted_idx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dterm))            

            # output hidden state corresponding to targetword
            try:
                term_hiddenstate_index = get_subidx(encoded_input.input_ids[0].tolist(), dterm_converted_idx)
            except ValueError:
                print(encoded_input.input_ids, dterm_converted_idx, text_seq)
                import pdb; pdb.set_trace()
                continue
            hiddenstates_dterm = output.hidden_states[0][0, term_hiddenstate_index: term_hiddenstate_index + len(dterm_converted_idx), :]
            hiddenstates_dterm_mean = np.mean(hiddenstates_dterm.detach().cpu().numpy(), axis=0, keepdims=False)

            hs = np.concatenate((hs, hiddenstates_dterm_mean))
        hidden_states.append(hs)
    return np.vstack(hidden_states)



 if __name__ == "__main__":

    # Load data
    train_path = '/data/users/yukatherin/hello/cs224u/data/wic/train.tsv'
    val_path = '/data/users/yukatherin/hello/cs224u/data/wic/dev.tsv'
    test_path = '/data/users/yukatherin/hello/cs224u/data/wic/test/test.data.txt'
    df_train = process_wic_data(train_path)
    df_val = process_wic_data(val_path)
    df_test = process_wic_data(test_path)
    print(df_train.shape)
    print(df_val.shape)
    print(df_test.shape)

    # Model
    bert_model = 'albert-base-v2'

    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModel.from_pretrained(bert_model)

    # Get the hidden states
    hs_train = process_hidden_states(df_train)
    hs_val = process_hidden_states(df_val)
    hs_test = process_hidden_states(df_test)
    joblib.dump((hs_train, df_train.label, hs_val, df_val.label, hs_test, df_test.label),'/data/users/yukatherin/hs.joblib')


    # Train MLP
    clf = MLPClassifier(random_state=1, hidden_layer_sizes=[20], alpha=1e-5, activation='relu', max_iter=300).fit(hs_train, df_train.label)

    # Collect metrics
    val_pred = clf.predict_proba(hs_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(df_val.label, val_pred)
    accuracy_scores = []
    for thresh in thresholds:
        try:
            pred_thresh = val_pred > thresh
            accuracy_scores.append(
                accuracy_score(df_val.label, pred_thresh)
            )
        except:
            import pdb; pdb.set_trace()
    accuracies = np.array(accuracy_scores)
    print(thresholds[accuracies.argmax()], accuracies.max(), f1_score(df_val.label, val_pred> thresholds[accuracies.argmax()]))

    # Run on test data
    preds_test = clf.predict_proba(hs_test)[:, 1] > thresholds[accuracies.argmax()]
    preds_test = preds_test.astype('uint8')
    index2class = {1:'T', 0:'F'}
    path_to_output_file_label = 'results/output_lab_{}.txt'.format('1')
    with open(path_to_output_file_label, 'w') as f:
        for val in preds_test:
            f.write("{}\n".format(index2class[val]))
