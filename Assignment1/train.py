import os
import torch
import torch.nn as nn
import tree
import senreps
import numpy as np
from init import calc_uuas, calc_fake_uuas, init_corpus, edges, parse_corpus, create_mst, create_gold_distances
from strucprobe import StructuralProbe, L1DistanceLoss, TwoWordBilinearLabelProbe, CrossEntropyLoss, make_fake_labels
from tree import print_tikz
from torch import Tensor
from transformers import *
from collections import defaultdict
from model import RNNModel
from torch import optim
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

def create_latex_tree(data, probe, index, corpus):
    '''
    Creates a latex tree at the given index with the printed uuas score
    '''
    pred = probe(data[1][index].unsqueeze(dim=0).to(device))
    labels = data[0][index]
    pred_mst = create_mst(pred[0])
    gold_mst = create_mst(labels)
    pred_edges = edges(pred_mst)
    gold_edges = edges(gold_mst)
    words = [word['form'] for word in corpus[index]]
    print_tikz(pred_edges, gold_edges, words)
    uuas_score = calc_uuas(pred[0], labels)
    print(uuas_score)

def load_models(model_type='TF'):
    '''
    Load transformers from huggingface when model_type='TF'
    Load LSTM from (gulordava et al, 2018) when model_type='RNN'
    '''
    if model_type == 'TF':
        model = GPT2Model.from_pretrained('distilgpt2').to(device=device)
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

    elif model_type == 'RNN':
        model_location = 'RNN/Gulordava.pt'
        model = torch.load(model_location, map_location=device)

        with open('RNN/vocab.txt') as f:
            w2i = {w.strip(): i for i, w in enumerate(f)}
            
        tokenizer = defaultdict(lambda: w2i["<unk>"])
        tokenizer.update(w2i)

    return model, tokenizer

def pad_labels(labels, seq_len, fake):
    '''
    Pad labels to match the corresponding max sentence length of predictions
    If fake=True the lables will have a dimensionality of 1 as opposed to 2
    '''
    padded_labels = []
    for label in labels:
        if fake:
            padded_label = torch.full((seq_len, 1), -1).squeeze()
            padded_label[:len(label)] = label
        else:
            padded_label = torch.full((seq_len, seq_len), -1)
            padded_label[:label.size(0), :label.size(1)] = label

        padded_labels.append(padded_label)

    return torch.stack(padded_labels)

def evaluate_probe(probe, data, loss_function, batch_size, fake):
    '''
    evaluation for both the dev and test set
    calculates the loss and the uuas score for the corresponding dataset
    '''
    loss_score_t = 0
    uuas_score = 0
    uuas_list = []

    for i in range(0, len(data[1]), batch_size):
        pred = probe(pad_sequence(data[1][i:i+batch_size], batch_first=True, padding_value=0).to(device))
        labels = pad_labels(data[0][i:i+batch_size], pred.size(1), fake).to(device)
        sen_len = data[2][i:i+batch_size].to(device)

        loss_score, _ = loss_function(pred, labels, sen_len)
        loss_score_t+=loss_score

        for j in range(len(pred)):
            pred_slice = pred[j][:int(sen_len[j]), :int(sen_len[j])]
            if fake:
                label_slice = labels[j][:int(sen_len[j])].to(dtype=torch.long)
                uuas_score += calc_fake_uuas(pred_slice, label_slice)
            else:
                label_slice = labels[j][:int(sen_len[j]), :int(sen_len[j])]
                cur_uuas_score = calc_uuas(pred_slice, label_slice)
                uuas_score += cur_uuas_score
                uuas_list.append((cur_uuas_score, sen_len[j]))

    loss_score_t/=len(data[1])
    uuas_score/=len(data[1])
        
    return loss_score, uuas_score, uuas_list

def train(data, rank, fake=False):
    '''
    Performs the training of the dependency tree task
    If fake=True then it will run as a control task
    Returns the test uuas score
    '''

    emb_dim = data['train'][1][0].size(1)
    lr = 10e-4
    batch_size = 24
    epochs = 50
    max_patience = 5
    patience = 0
    best_uuas = 0

    dev_corpus = parse_corpus('data/en_ewt-ud-dev.conllu')
    test_corpus = parse_corpus('data/en_ewt-ud-test.conllu')

    #If it is a control task, another probe is used whos output is adaptable to CE loss 
    if fake:
        probe = TwoWordBilinearLabelProbe(emb_dim, rank, 0.4, device=device)
        loss_function = CrossEntropyLoss()
    else:
        probe = StructuralProbe(emb_dim, rank, device=device)
        loss_function = L1DistanceLoss()
    
    probe = StructuralProbe(emb_dim, rank, device=device)
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

    #code for calculating uuas scores of different sentence lengths
    '''
    uuas_5 = []
    uuas_6_10 = []
    uuas_11_15 = []
    uuas_16 = []

    for uuas_score, len_sen in uuas_list:
        if len_sen<=5:
            uuas_5.append(uuas_score)
        elif len_sen<=10:
            uuas_6_10.append(uuas_score)
        elif len_sen<=15:
            uuas_11_15.append(uuas_score)
        else:
            uuas_16.append(uuas_score)

    print(np.mean(uuas_5), len(uuas_5))
    print(np.mean(uuas_6_10), len(uuas_6_10))
    print(np.mean(uuas_11_15), len(uuas_11_15))
    print(np.mean(uuas_16), len(uuas_16))
    '''

    print("\n---------------------------------------------------------")    
    print("test set loss: " + str(test_loss.item()) + ", test-uuas: " + str(test_uuas))
    print("---------------------------------------------------------")

    return test_uuas

if __name__ == '__main__':
    control = False

    data = {}
    model_file = 'TF_GPT2-distil.pt'
    model_type = 'TF'
    gold_file = 'gold_data.pt'
    fake_gold_file = 'fake_gold_data.pt'

    print('load data')
    if not os.path.exists(model_file):
        model, tokenizer = load_models(model_type=model_type)
        data_model = {}
        data_model['train'] = init_corpus('data/en_ewt-ud-train.conllu', model, tokenizer, model_type, device)
        data_model['dev'] = init_corpus('data/en_ewt-ud-dev.conllu', model, tokenizer, model_type, device)
        data_model['test'] = init_corpus('data/en_ewt-ud-test.conllu', model, tokenizer, model_type, device)
        torch.save(data_model, model_file)
    else:
        data_model = torch.load(model_file)

    #rankings = [64]
    rankings = [1, 2, 4, 8, 16, 32, 64, 128]

    #load real golden distance trees
    if not control:
        if not os.path.exists(gold_file):
            gold_data = {}
            gold_data['train'] = create_gold_distances(parse_corpus('data/en_ewt-ud-train.conllu'))
            gold_data['dev'] = create_gold_distances(parse_corpus('data/en_ewt-ud-dev.conllu'))
            gold_data['test'] = create_gold_distances(parse_corpus('data/en_ewt-ud-test.conllu'))
            torch.save(gold_data, gold_file)
        else:
            gold_data = torch.load(gold_file)
    
    #load fake golden distance trees
    else:
        if not os.path.exists(fake_gold_file):
            gold_data = {}
            gold_data['train'], fake_vocab = make_fake_labels(parse_corpus('data/en_ewt-ud-train.conllu'), fake_vocab=None)
            gold_data['dev'] = make_fake_labels(parse_corpus('data/en_ewt-ud-dev.conllu'), fake_vocab=fake_vocab)[0]
            gold_data['test'] = make_fake_labels(parse_corpus('data/en_ewt-ud-test.conllu'), fake_vocab=fake_vocab)[0]
            torch.save(gold_data, fake_gold_file)
        else:
            gold_data = torch.load(fake_gold_file)

    for key in gold_data:
        data[key] = [gold_data[key], data_model[key][0], data_model[key][1]]

    print('begin training')
    uuas_scores = []

    for rank in rankings:
        print('rank: ' + str(rank))
        uuas_scores.append(train(data, rank, fake=control))
    
    print(uuas_scores)
    torch.save(uuas_scores, 'uuas-'+model_file)

    


