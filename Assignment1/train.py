import torch
import torch.nn as nn
import tree
import senreps
import numpy as np
from init import calc_uuas, init_corpus, edges, parse_corpus, create_mst
from strucprobe import StructuralProbe, L1DistanceLoss
from tree import print_tikz
from torch import Tensor
from transformers import *
from collections import defaultdict
from model import RNNModel
from torch import optim
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_latex_tree(data, probe, index, corpus):
    pred = probe(data[1][index].unsqueeze(dim=0).to(device))
    labels = data[0][index]
    pred_mst = create_mst(pred[0])
    gold_mst = create_mst(labels)
    pred_edges = edges(pred_mst)
    gold_edges = edges(gold_mst)
    words = [word['form'] for word in corpus[index]]
    print_tikz(pred_edges, gold_edges, words)
    print('uuas score:' + str(calc_uuas(pred[0], labels)))

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

def pad_labels(labels, seq_len):
    padded_labels = []
    for label in labels:
        padded_label = torch.full((seq_len, seq_len), -1)
        padded_label[:label.size(0), :label.size(1)] = label
        padded_labels.append(padded_label)

    return torch.stack(padded_labels)

        
# I recommend you to write a method that can evaluate the UUAS & loss score for the dev (& test) corpus.
# Feel free to alter the signature of this method.
def evaluate_probe(probe, data, loss_function, batch_size):
    loss_score_t = 0
    total_sents_t = 0
    uuas_score = 0

    for i in range(0, len(data[1]), batch_size):
        pred = probe(pad_sequence(data[1][i:i+batch_size], batch_first=True, padding_value=0).to(device))
        labels = pad_labels(data[0][i:i+batch_size], pred.size(1)).to(device)
        sen_len = data[2][i:i+batch_size].to(device)

        loss_score, total_sents = loss_function(pred, labels, sen_len)
        loss_score_t+=loss_score

        for j in range(len(pred)):
            pred_slice = pred[j][:int(sen_len[j]), :int(sen_len[j])]
            label_slice = labels[j][:int(sen_len[j]), :int(sen_len[j])]
            uuas_score += calc_uuas(pred_slice, label_slice)

    loss_score_t/=len(data[1])
    uuas_score/=len(data[1])
        
    return loss_score, uuas_score


# Feel free to alter the signature of this method.
def train(data):
    emb_dim = data['train'][1][0].size(1)
    rank = 64
    lr = 10e-4
    batch_size = 24
    epochs = 50

    dev_corpus = parse_corpus('data/en_ewt-ud-dev.conllu')
    test_corpus = parse_corpus('data/en_ewt-ud-test.conllu')
    probe = StructuralProbe(emb_dim, rank, device=device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=1)
    loss_function =  L1DistanceLoss()
        
    for epoch in range(epochs):
        print("\n---------------------------------------------------------") 
        print("epoch: " + str(epoch+1))
        print("---------------------------------------------------------") 
        
        print("\n---------------------------------------------------------")
        batch_loss_total = []
        for i in range(0, len(data['train'][1]), batch_size):
            optimizer.zero_grad()

            # YOUR CODE FOR DOING A PROBE FORWARD PASS
            pred = probe(pad_sequence(data['train'][1][i:i+batch_size], batch_first=True, padding_value=0).to(device))
            labels = data['train'][0][i:i+batch_size]
            labels = pad_labels(labels, pred.size(1)).to(device)
            sen_len = data['train'][2][i:i+batch_size].to(device)
            
            batch_loss, total_sents = loss_function(pred, labels, sen_len)
            batch_loss_total.append(batch_loss.item())
            
            if i%(batch_size*50)==0:
                print("train set loss: " + str(np.mean(batch_loss_total)) + ", iter: " + str(i) + '/' + str(len(data['train'][1])))
                batch_loss_total = []
            
            batch_loss.backward()
            optimizer.step()

        dev_loss, dev_uuas = evaluate_probe(probe, data['dev'], loss_function, batch_size)
        
        print("\n---------------------------------------------------------")    
        print("dev set loss: " + str(dev_loss.item()) + ", dev-uuas: " + str(dev_uuas))
        print("---------------------------------------------------------")        
        

        # Using a scheduler is up to you, and might require some hyper param fine-tuning
        scheduler.step(dev_loss)

    total = 0
    for i in range(len(test_corpus)):
        if len(test_corpus[i])<=10:
            create_latex_tree(data['test'], probe, i, test_corpus)
            total+=1
        if total==5:
            break

    test_loss, test_uuas = evaluate_probe(probe, data['test'], loss_function, batch_size)
    
    print("\n---------------------------------------------------------")    
    print("test set loss: " + str(test_loss.item()) + ", test-uuas: " + str(test_uuas))
    print("---------------------------------------------------------")

if __name__ == '__main__':
    data = {}
    '''

    model, tokenizer = load_models(model_type='TF')
    print('model loaded')
    print('load data')
    
    data['train'] = init_corpus('data/en_ewt-ud-train.conllu', model, tokenizer, 'TF', device)
    data['dev'] = init_corpus('data/en_ewt-ud-dev.conllu', model, tokenizer, 'TF', device)
    data['test'] = init_corpus('data/en_ewt-ud-test.conllu', model, tokenizer, 'TF', device)
    torch.save(data, 'TF_XLNet.pt')
    '''

    gold_data = torch.load('gold_data.pt')
    data_model = torch.load('TF_XLNet.pt')

    for key in gold_data:
        data[key] = [gold_data[key], data_model[key][0], data_model[key][1]]

    print('begin training')

    train(data)

    


