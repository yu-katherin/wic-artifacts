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
import joblib

from sklearn.metrics import accuracy_score, roc_curve, f1_score
from dataset import process_wic_data, CustomDataset
from model_utils import *  # sorry!


os.environ["TOKENIZERS_PARALLELISM"] = "false"


my_experiments = {
    '1a': MyExperiment(
        freeze_bert = False,
        iters_to_accumulate = 2,
        lr = 2e-5,
        epochs = 3,
    ),
    '1b': MyExperiment(
        freeze_bert = True,
        iters_to_accumulate = 2,
        lr = 2e-5,
        epochs = 6,
    ),
    '2a-': MyExperiment(
        freeze_bert = False,
        iters_to_accumulate = 2,
        lr = 2e-5,
        epochs = 3,
        mask_sent1=True,
    ),
    '2b-': MyExperiment(
        freeze_bert = True,
        iters_to_accumulate = 2,
        lr = 2e-5,
        epochs = 6,
        mask_sent1=True,
    ),
    '3a-': MyExperiment(
        freeze_bert = False,
        iters_to_accumulate = 2,
        lr = 2e-5,
        epochs = 3,
        mask_sent2=True,
    ),
    '3b-': MyExperiment(
        freeze_bert = True,
        iters_to_accumulate = 2,
        lr = 2e-5,
        epochs = 6,
        mask_sent2=True,
    ),
}


if __name__ == '__main__':

    # Load raw data
    train_path = '/data/users/yukatherin/hello/cs224u/data/wic/train.tsv'
    val_path = '/data/users/yukatherin/hello/cs224u/data/wic/dev.tsv'
    test_path = '/data/users/yukatherin/hello/cs224u/data/wic/test/test.data.txt'


    ## Parameters
    bert_model = "albert-base-v2"


    #  Set all seeds to make reproducible results
    set_seed(1)
    # Creating instances of training and validation set
    bs, maxlen = 16, 128


    # Run experiments
    for experimentname, params in my_experiments.items():

        df_train = process_wic_data(train_path, mask_sent1=params.mask_sent1, mask_sent2=params.mask_sent2)
        df_val = process_wic_data(val_path, mask_sent1=params.mask_sent1, mask_sent2=params.mask_sent2)
        df_test = process_wic_data(test_path, is_test=True, mask_sent1=params.mask_sent1, mask_sent2=params.mask_sent2)
        print(df_train.shape)
        print(df_val.shape)
        print(df_test.shape)

        print("Reading training data...")
        train_set = CustomDataset(df_train, maxlen=maxlen, bert_model=bert_model, with_labels=True)
        print("Reading validation data...")
        val_set = CustomDataset(df_val, maxlen=maxlen, bert_model=bert_model, with_labels=True)
        # Creating instances of training and validation dataloaders
        train_loader = DataLoader(train_set, batch_size=bs, num_workers=5)
        val_loader = DataLoader(val_set, batch_size=bs, num_workers=5)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net = SentencePairClassifier(bert_model, freeze_bert=params.freeze_bert)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)

        net.to(device)

        criterion = nn.BCEWithLogitsLoss()
        opti = AdamW(net.parameters(), lr=params.lr, weight_decay=1e-2)
        num_warmup_steps = 0
        num_training_steps = params.epochs * len(train_loader)
        t_total = (len(train_loader) // params.iters_to_accumulate) * params.epochs
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

        path_to_model, best_thres = train_bert(
            net=net,
            criterion=criterion,
            opti=opti,
            lr=params.lr,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=params.epochs,
            iters_to_accumulate=params.iters_to_accumulate,
            experimentname=experimentname,
        )
        # path_to_model = 'models/albert-base-v2_2a-_acc_0.5971786833855799_f1_0.5965463108320251_thres_0.6435546875_best.pt'

        path_to_output_file = 'results/output_{}.txt'.format(experimentname)
        path_to_output_file_label = 'results/output_lab_{}.txt'.format(experimentname)

        print("Reading test data...")
        test_set = CustomDataset(df_test, maxlen=maxlen, bert_model=bert_model, with_labels=False)
        test_loader = DataLoader(test_set, batch_size=bs, num_workers=5)

        print("Loading the weights of the model...")
        model = SentencePairClassifier(bert_model)
        if torch.cuda.device_count() > 1:  # if multiple GPUs
            model = nn.DataParallel(model)
        model.load_state_dict(torch.load(path_to_model))
        model.to(device)

        print("Predicting on test data {}".format(path_to_output_file_label))
        test_prediction(net=model, device=device, dataloader=test_loader, with_labels=False, result_file=path_to_output_file)
        probs_test = pd.read_csv(path_to_output_file, header=None)[0]  # prediction probabilities
        preds_test=(probs_test>=best_thres).astype('uint8') # predicted labels using the above fixed threshold
        index2class = {1:'T', 0:'F'}
        with open(path_to_output_file_label, 'w') as f:
            for val in preds_test:
                f.write("{}\n".format(index2class[val]))

        print("Computing confidence interval for validation accuracy")
        val_pred = test_prediction(net=net, device=device, dataloader=val_loader, with_labels=True, result_file='results/output_bestval.txt')
        val_pred = np.array(val_pred)
        acc_samples = []
        bootstrap_samples = 10000
        sample_size = 0.8
        for i in range(bootstrap_samples):
            if i % 200 == 0:
                print(i)
            np.random.seed(seed=i)
            sample_indices = np.random.choice(df_val.shape[0], int(sample_size * df_val.shape[0]))
            pred_bs, label_bs = val_pred[sample_indices], df_val.label[sample_indices]
            fpr, tpr, thresholds = roc_curve(label_bs, pred_bs)
            accuracy_scores = []
            for thresh in thresholds:
                pred_thresh = pred_bs > thresh
                accuracy_scores.append(
                    accuracy_score(label_bs, pred_thresh)
                )
            best_acc = np.array(accuracy_scores).max()
            acc_samples.append(best_acc)
        try:
            joblib.dump(acc_samples, "results/{}_acc_samples.joblib".format(experimentname))
        except:
            import pdb; pdb.set_trace()
