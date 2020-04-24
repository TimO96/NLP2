import os
import torch
import torch.nn as nn
import tree
import senreps
import numpy as np
from init import create_data, parse_corpus
from DC import DC
from senreps import fetch_fake_pos_tags
from torch import Tensor
from transformers import *
from collections import defaultdict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

def load_models(model_type='TF'):
    '''
    Load transformers from huggingface when model_type='TF'
    Load LSTM from (gulordava et al, 2018) when model_type='RNN'
    '''
    if model_type == 'TF':
        model = GPT2Model.from_pretrained('distilgpt2').to(device=device)
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

    elif model_type == 'RNN':
        model_location = 'RNN/Gulordava.pt'  # <- point this to the location of the Gulordava .pt file
        model = torch.load(model_location, map_location=device)

        with open('RNN/vocab.txt') as f:
            w2i = {w.strip(): i for i, w in enumerate(f)}
            
        tokenizer = defaultdict(lambda: w2i["<unk>"])
        tokenizer.update(w2i)

    return model, tokenizer

def evaluate_pos(test_x, test_y, model_DC, total_batches_test, batch_size):
    '''
    evaluation for both the dev and test set
    calculates the accuracy for the corresponding dataset
    '''
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
    '''
    Performs the training of the POS-tagging task
    Prints the accuracy that is obtained from the test set
    '''

    learning_rate = 1e-3
    batch_size = 64
    epochs = 500   

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
    control = False

    data = {}
    model_file = 'TF_GPT2-distil.pt'
    model_type = 'TF'
    fake_file = 'fake-pos.pt'
    
    print('load data')
    if not os.path.exists('pos-'+model_file):
        model, tokenizer = load_models(model_type=model_type)
        data['train'] = create_data('data/en_ewt-ud-train.conllu', model, tokenizer, model_type, device)
        data['dev'] = create_data('data/en_ewt-ud-dev.conllu', model, tokenizer, model_type, device, vocab=data['train'][2])
        data['test'] = create_data('data/en_ewt-ud-test.conllu', model, tokenizer, model_type, device, vocab=data['train'][2])
        torch.save(data, 'pos-'+model_file)
    else:
        data = torch.load('pos-'+model_file)

    if control:
        if not os.path.exists(fake_file):
            fake_train_pos, fake_vocab = fetch_fake_pos_tags(parse_corpus('data/en_ewt-ud-train.conllu'), data['train'][2], fake_vocab=None)
            fake_dev_pos = fetch_fake_pos_tags(parse_corpus('data/en_ewt-ud-dev.conllu'), data['train'][2], fake_vocab=fake_vocab)[0]
            fake_test_pos = fetch_fake_pos_tags(parse_corpus('data/en_ewt-ud-test.conllu'), data['train'][2], fake_vocab=fake_vocab)[0]
            torch.save([fake_train_pos, fake_dev_pos, fake_test_pos], fake_file)
        else:
            fake_train_pos, fake_dev_pos, fake_test_pos = torch.load(fake_file)
    
        data['train'] = (data['train'][0], fake_train_pos, data['train'][2])
        data['dev'] = (data['dev'][0], fake_dev_pos, data['dev'][2])
        data['test'] = (data['test'][0], fake_test_pos, data['test'][2])

    print('data loaded')

    print('begin training')

    train(data)
