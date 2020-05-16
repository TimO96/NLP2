import os
import torch
import torch.nn as nn
import numpy as np
from dataloaders import EmbeddingDataset
from torch import Tensor
from transformers import *
from torch import optim
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

def train(data, epochs=10, batch_size=64, lr=1e-3, early_stopping=False):

    emb_dim = data['train'][0][0].size(1)
    max_patience = 5
    patience = 0

    train_data = DataLoader(EmbeddingDataset(data['train']), batch_size=batch_size, shuffle=True, num_workers=4)
    dev_data = DataLoader(EmbeddingDataset(data['dev']), batch_size=batch_size, shuffle=True, num_workers=4)
    test_data = DataLoader(EmbeddingDataset(data['test']), batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam(probe.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=1)
        
    for epoch in range(epochs):
        print("\n---------------------------------------------------------") 
        print("epoch: " + str(epoch+1))
        print("---------------------------------------------------------") 
        
        print("\n---------------------------------------------------------")
        batch_loss_total = []
        for i in range(0, len(data['train'][1]), batch_size):
            optimizer.zero_grad()

            pred = probe(pad_sequence(data['train'][1][i:i+batch_size], batch_first=True, padding_value=0).to(device))
            labels = data['train'][0][i:i+batch_size]
            labels = pad_labels(labels, pred.size(1), fake).to(device)
            sen_len = data['train'][2][i:i+batch_size].to(device)
      
            batch_loss, total_sents = loss_function(pred, labels, sen_len)
            batch_loss_total.append(batch_loss.item())
            
            if i%(batch_size*50)==0:
                print("train set loss: " + str(np.mean(batch_loss_total)) + ", iter: " + str(i) + '/' + str(len(data['train'][1])))
                batch_loss_total = []
            
            batch_loss.backward()
            optimizer.step()

        dev_loss, dev_uuas, _ = evaluate_probe(probe, data['dev'], loss_function, batch_size, fake)

        #Early stopping implemented on uuas of dev dataset
        if dev_uuas>best_uuas:
            best_uuas = dev_uuas
            patience = 0
        
        else:
            patience+=1
            if max_patience==patience:
                print("\n---------------------------------------------------------")    
                print("dev set loss: " + str(dev_loss.item()) + ", dev-uuas: " + str(dev_uuas) + " patience: " + str(patience))
                print("early stopping")
                print("---------------------------------------------------------") 
                break

        print("\n---------------------------------------------------------")    
        print("dev set loss: " + str(dev_loss.item()) + ", dev-uuas: " + str(dev_uuas) + " patience: " + str(patience))
        print("---------------------------------------------------------")         

        scheduler.step(dev_loss)

    test_loss, test_uuas, uuas_list = evaluate_probe(probe, data['test'], loss_function, batch_size, fake)

    print("\n---------------------------------------------------------")    
    print("test set loss: " + str(test_loss.item()) + ", test-uuas: " + str(test_uuas))
    print("---------------------------------------------------------")

    return test_uuas