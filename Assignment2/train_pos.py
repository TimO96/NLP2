import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from polyglot.mapping import Embedding
from polyglot.downloader import downloader
from senreps import fetch_pos_tags
from DC import DC
from torch import Tensor
from transformers import *
from collections import defaultdict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.manual_seed(42)
#np.random.seed(42)

def evaluate_pos(test_x, test_y, model_DC, total_batches_test, batch_size, proj, W):
    '''
    evaluation for both the dev and test set
    calculates the accuracy for the corresponding dataset
    '''
    correct = 0
    indices = torch.randperm(len(test_x))
    for i in range(total_batches_test):
        index = indices[i*batch_size:(i+1)*batch_size]
        features = test_x[index].to(dtype=torch.float, device=device)
        labels = test_y[index].to(dtype=torch.long, device=device)
        
        if proj:
            features = torch.matmul(features, W)

        preds = model_DC(features)
        correct += torch.sum((preds.max(1)[1] == labels).int()).item()

    acc_test = correct/len(test_x)
    print("test set accuracy: " + str(acc_test))
    print("---------------------------------------------------------")

    return acc_test


def train(data_x, data_y, W=None, proj=True):
    '''
    Performs the training of the POS-tagging task
    Returns the accuracy that is obtained from the test set
    '''

    learning_rate = 1e-3
    batch_size = 64
    epochs = 100
    embed_size = 768 

    loss = nn.CrossEntropyLoss()

    train_x = data_x['train']
    dev_x = data_x['dev']
    test_x = data_x['test']

    train_y = data_y['train'][0]
    vocab = data_y['train'][1]
    dev_y = data_y['dev'][0]
    test_y = data_y['test'][0]

    #Diagnostic classifier used to map embeddings -> POS-tags
    model_DC = DC(embed_size, len(vocab)).to(device)
    optimizer = torch.optim.Adam(model_DC.parameters(), lr=learning_rate)

    total_batches = int(np.ceil(len(train_x)/batch_size))
    total_batches_dev = int(np.ceil(len(dev_x)/batch_size))
    total_batches_test = int(np.ceil(len(test_x)/batch_size))
    min_loss = np.inf
    patience = 0
    max_patience = 5
    
    for epoch in range(epochs):
        losses = []
        indices = torch.randperm(len(train_x))

        for i in range(total_batches):
            index = indices[i*batch_size:(i+1)*batch_size]
            features = train_x[index].to(dtype=torch.float, device=device)
            labels = train_y[index].to(dtype=torch.long, device=device)

            if proj:
                features = torch.matmul(features, W)

            preds = model_DC(features)
            loss_out = loss(preds, labels)

            optimizer.zero_grad()
            loss_out.backward()
            optimizer.step()
            losses.append(loss_out.item())
            
        loss_train = np.mean(losses)
        
        print("\n---------------------------------------------------------")
        print("epoch: " + str(epoch+1))
        print("training set loss: " + str(loss_train))
        
        with torch.no_grad():
            losses = []
            indices = torch.randperm(len(dev_x))

            for i in range(total_batches_dev):
                index = indices[i*batch_size:(i+1)*batch_size]
                features = dev_x[index].to(dtype=torch.float, device=device)
                labels = dev_y[index].to(dtype=torch.long, device=device)
                
                if proj:
                    features = torch.matmul(features, W)

                preds = model_DC(features)
                loss_out = loss(preds, labels)
                losses.append(loss_out.item())
            
            loss_dev = np.mean(losses)
            
            #Early Stopping
            if loss_dev>min_loss:
                patience+=1
            else:
                min_loss = loss_dev
                patience = 0

            print("validation set loss: " + str(loss_dev) + ", patience: " + str(patience))
            print("---------------------------------------------------------")

            if patience==max_patience:
                print("\n---------------------------------------------------------")
                print("Early Stopping at epoch:", str(epoch+1))
                break

    acc_test = evaluate_pos(test_x, test_y, model_DC, total_batches_test, batch_size, proj, W) 

    return acc_test


