import os
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from transformers import *
from torch import optim
from torch.nn.init import xavier_uniform_, xavier_normal_

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.manual_seed(42)
#np.random.seed(42)

def train(data, epochs=100, batch_size=64, lr=1e-4, early_stopping=False, constraint=False):
    '''
    Optimizes the projection matrix W
    when constraint = False -> Mikolov (2013) method
    when constraint = True -> Xing (2015) method
    Returns a projection matrix W
    '''

    source_dim = data['train'][0][0].size(1)
    target_dim = data['train'][1][0].size(1)
    max_patience = 5
    patience = 0
    best_loss = np.inf
    loss_threshold = 1e-2
    weight_c = 100

    train_data = data['train']
    dev_data = data['dev']
    test_data = data['test']

    loss_func = nn.MSELoss()
    W = xavier_uniform_(torch.empty(source_dim, target_dim)).requires_grad_(True) 
    optimizer = optim.Adam([W], lr=lr)

    for epoch in range(epochs):
        print("\n---------------------------------------------------------") 
        print("epoch: " + str(epoch+1))
        print("---------------------------------------------------------") 
        
        print("\n---------------------------------------------------------")
        batch_loss = []
        indices = torch.randperm(len(train_data[0]))
        batch_length = int(np.ceil(len(train_data[0])/batch_size))

        for i in range(batch_length):
            index = indices[i*batch_size:(i+1)*batch_size]
            features = torch.cat([train_data[0][i] for i in index])
            labels = torch.cat([train_data[1][i] for i in index])

            optimizer.zero_grad()

            preds = torch.matmul(features, W)

            if constraint:
                u, s, v = torch.svd(torch.matmul(features.transpose(0, 1), labels))
                W_constr = torch.matmul(u, v.transpose(0, 1))
                loss = loss_func(W, W_constr)*weight_c

            else:
                loss = loss_func(preds, labels)

            batch_loss.append(loss.item())
            
            if i%50==0:
                print("train set loss: " + str(np.mean(batch_loss)) + ", iter: " + str(i*batch_size) + '/' + str(len(indices)))
                batch_loss = []
            
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            indices = torch.randperm(len(dev_data[0]))
            batch_length = int(np.ceil(len(dev_data[0])/batch_size))
            batch_loss = []

            for i in range(batch_length):
                index = indices[i*batch_size:(i+1)*batch_size]
                features = torch.cat([dev_data[0][i] for i in index])
                labels = torch.cat([dev_data[1][i] for i in index])

                preds = torch.matmul(features, W)

                if constraint:
                    u, s, v = torch.svd(torch.matmul(features.transpose(0, 1), labels))
                    W_constr = torch.matmul(u, v.transpose(0, 1))
                    loss = loss_func(W, W_constr)*weight_c

                else:
                    loss = loss_func(preds, labels)
                
                batch_loss.append(loss.item())
            
        mean_loss = np.mean(batch_loss)

        #Early stopping
        if mean_loss<best_loss:
            if best_loss-mean_loss<best_loss*loss_threshold:
                patience+=1
                if max_patience==patience:
                    print("\n---------------------------------------------------------")    
                    print("dev loss: " + str(mean_loss) + " patience: " + str(patience))
                    print("early stopping")
                    print("---------------------------------------------------------") 
                    break
            else:
                patience = 0

            best_loss = mean_loss
        
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

    with torch.no_grad():
        indices = torch.randperm(len(test_data[0]))
        batch_length = int(np.ceil(len(test_data[0])/batch_size))
        batch_loss = []

        for i in range(batch_length):
            index = indices[i*batch_size:(i+1)*batch_size]
            features = torch.cat([test_data[0][i] for i in index])
            labels = torch.cat([test_data[1][i] for i in index])

            preds = torch.matmul(features, W)

            if constraint:
                u, s, v = torch.svd(torch.matmul(features.transpose(0, 1), labels))
                W_constr = torch.matmul(u, v.transpose(0, 1))
                loss = loss_func(W, W_constr)*weight_c

            else:
                loss = loss_func(preds, labels)

            batch_loss.append(loss.item())

    mean_loss = np.mean(batch_loss)

    print("\n---------------------------------------------------------")    
    print("test loss: " + str(mean_loss))
    print("---------------------------------------------------------")

    return W