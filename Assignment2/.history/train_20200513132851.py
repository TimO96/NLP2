import os
import torch
import torch.nn as nn
import numpy as np
from dataloaders import EmbeddingDataset
from torch import Tensor
from transformers import *
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.init import xavier_uniform_

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

def train(data, epochs=10, batch_size=64, lr=1e-4, early_stopping=False):

    source_dim = data['train'][0][0].size(1)
    target_dim = data['train'][1][0].size(1)
    max_patience = 5
    patience = 0
    best_loss = np.inf

    train_data = data['train']
    dev_data = data['dev']
    test_data = data['test']

    loss_func = nn.MSELoss()
    W = xavier_uniform_(torch.empty(source_dim, target_dim)) 
    optimizer = optim.Adam([W], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=1)   

    for epoch in range(epochs):
        print("\n---------------------------------------------------------") 
        print("epoch: " + str(epoch+1))
        print("---------------------------------------------------------") 
        
        print("\n---------------------------------------------------------")
        batch_loss = []
        indices = torch.randperm(len(train_data[0]))

        for i in indices:
            features = train_data[0][i]
            labels = train_data[1][i]

            optimizer.zero_grad()

            preds = torch.matmul(features, W)

            loss = loss_func(preds, labels)

            print(preds.size())
            print(labels.size())
            print(loss)
      
            batch_loss.append(loss.item())
            
            if i%50==0:
                print("train set loss: " + str(np.mean(batch_loss)) + ", iter: " + str(i*batch_size) + '/' + str(len(train_data)))
                batch_loss = []
            
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            indices = torch.randperm(len(dev_data[0]))
            batch_loss = []

            for i in indices:
                features = dev_data[0][i]
                labels = dev_data[1][i]

                preds = torch.matmul(features, W)

                loss = loss_func(preds, labels)
                batch_loss.append(loss.item())
            
        mean_loss = np.mean(batch_loss)

        #Early stopping implemented on uuas of dev dataset
        if mean_loss<best_loss:
            best_loss = mean_loss
            patience = 0
        
        else:
            patience+=1
            if max_patience==patience:
                print("\n---------------------------------------------------------")    
                print("dev loss: " + str(mean_loss) + " patience: " + str(patience))
                print("early stopping")
                print("---------------------------------------------------------") 
                break

        print("\n---------------------------------------------------------")    
        print("dev loss: " + str(mean_loss) + " patience: " + str(patience))
        print("---------------------------------------------------------")         

        scheduler.step(loss)

    with torch.no_grad():
        indices = torch.randperm(len(test_data[0]))
        batch_loss = []

        for i in indices:
            features = test_data[0][i]
            labels = test_data[1][i]

            preds = torch.matmul(features, W)

            loss = loss_func(preds, labels)
            batch_loss.append(loss.item())

    mean_loss = np.mean(batch_loss)

    print("\n---------------------------------------------------------")    
    print("test loss: " + str(mean_loss))
    print("---------------------------------------------------------")

    return W

if __name__ == "__main__":
    data = torch.load('xlingual_data.pt')
    train(data, early_stopping=True)