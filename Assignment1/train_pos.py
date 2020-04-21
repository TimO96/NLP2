import torch
import torch.nn as nn
import tree
import senreps
import numpy as np
from init import create_data
from DC import DC
from torch import Tensor
from transformers import *
from collections import defaultdict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_models(model_type='TF'):
    if model_type == 'TF':
        model = XLNetModel.from_pretrained('xlnet-base-cased').to(device=device)
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    elif model_type == 'RNN':
        model_location = 'RNN/Gulordava.pt'  # <- point this to the location of the Gulordava .pt file
        model = torch.load(model_location, map_location=device)

        with open('RNN/vocab.txt') as f:
            w2i = {w.strip(): i for i, w in enumerate(f)}
            
        tokenizer = defaultdict(lambda: w2i["<unk>"])
        tokenizer.update(w2i)

    return model, tokenizer

def evaluate_pos(test_x, test_y, model_DC, total_batches_test, batch_size):
    correct = 0
    for i in range(total_batches_test):
        if i < total_batches_test-1:
            features = test_x[i*batch_size:(i+1)*batch_size].to(dtype=torch.float, device=device)
            labels = test_y[i*batch_size:(i+1)*batch_size].to(dtype=torch.long, device=device)
        else:
            features = test_x[i*batch_size:].to(dtype=torch.float, device=device)
            labels = test_y[i*batch_size:].to(dtype=torch.long, device=device)
        
        preds = model_DC(features)
        correct += torch.sum((preds.max(1)[1] == labels).int()).item()

    acc_test = correct/len(test_x)
    print("test set accuracy: " + str(acc_test))
    print("---------------------------------------------------------")


def train(data):
    learning_rate = 1e-3
    batch_size = 64
    epochs = 100   

    loss = nn.CrossEntropyLoss()

    train_x = data['train'][0]
    train_y = data['train'][1]
    vocab = data['train'][2]
    dev_x = data['dev'][0]
    dev_y = data['dev'][1]
    test_x = data['test'][0]
    test_y = data['test'][1]

    model_DC = DC(train_x.size(1), len(vocab)).to(device)
    optimizer = torch.optim.Adam(model_DC.parameters(), lr=learning_rate)
    
    total_batches = int(len(train_x)/batch_size)+1    
    total_batches_dev = int(len(dev_x)/batch_size)+1
    total_batches_test = int(len(test_x)/batch_size)+1
    min_loss = np.inf
    patience = 0
    max_patience = 5
    
    for epoch in range(epochs):
        losses = []
        for i in range(total_batches):
            if i < total_batches-1:
                features = train_x[i*batch_size:(i+1)*batch_size].to(dtype=torch.float, device=device)
                labels = train_y[i*batch_size:(i+1)*batch_size].to(dtype=torch.long, device=device)
            else:
                features = train_x[i*batch_size:].to(dtype=torch.float, device=device)
                labels = train_y[i*batch_size:].to(dtype=torch.long, device=device)


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
            for i in range(total_batches_dev):
                if i < total_batches_dev-1:
                    features = dev_x[i*batch_size:(i+1)*batch_size].to(dtype=torch.float, device=device)
                    labels = dev_y[i*batch_size:(i+1)*batch_size].to(dtype=torch.long, device=device)
                else:
                    features = dev_x[i*batch_size:].to(dtype=torch.float, device=device)
                    labels = dev_y[i*batch_size:].to(dtype=torch.long, device=device)

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

    evaluate_pos(test_x, test_y, model_DC, total_batches_test, batch_size)            
        

if __name__ == '__main__':
    data = {}
    model_type = 'TF'

    model, tokenizer = load_models(model_type=model_type)
    print('load data')
    
    data['train'] = create_data('data/en_ewt-ud-train.conllu', model, tokenizer, model_type, device)
    data['dev'] = create_data('data/en_ewt-ud-dev.conllu', model, tokenizer, model_type, device, vocab=data['train'][2])
    data['test'] = create_data('data/en_ewt-ud-test.conllu', model, tokenizer, model_type, device, vocab=data['train'][2])
    torch.save(data, model_type+'-XLNet-pos.pt')

    data = torch.load(model_type+'-XLNet-pos.pt')
    print('model loaded')

    print('begin training')

    
    train(data)
