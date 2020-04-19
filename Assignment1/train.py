import torch
import torch.nn as nn
import tree
import senreps
import numpy as np
from init import calc_uuas, init_corpus
from strucprobe import StructuralProbe, L1DistanceLoss
from torch import Tensor
from transformers import *
from collections import defaultdict
from model import RNNModel
from torch import optim
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_models(model_type='TF'):
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
        total_sents_t+=total_sents

        for i in range(len(pred)):
            pred_slice = pred[i][:int(sen_len[i]), :int(sen_len[i])]
            label_slice = labels[i][:int(sen_len[i]), :int(sen_len[i])]
            uuas_score += calc_uuas(pred_slice, label_slice)

    loss_score_t/=total_sents_t
    uuas_score/=len(data[1])
        
    return loss_score, uuas_score


# Feel free to alter the signature of this method.
def train(data):
    emb_dim = 768
    rank = 64
    lr = 10e-4
    batch_size = 24
    epochs = 10

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
            batch_loss/=total_sents
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

    test_loss, test_uuas = evaluate_probe(probe, data['test'], loss_function, batch_size)
    
    print("\n---------------------------------------------------------")    
    print("test set loss: " + str(test_loss.item()) + ", test-uuas: " + str(test_uuas))
    print("---------------------------------------------------------")

if __name__ == '__main__':
    data = {}

    model, tokenizer = load_models(model_type='TF')
    print('model loaded')
    print('load data')
    #data['train'] = init_corpus('data/en_ewt-ud-train.conllu', model, tokenizer, 'TF', device)
    #data['dev'] = init_corpus('data/en_ewt-ud-dev.conllu', model, tokenizer, 'TF', device)
    #data['test'] = init_corpus('data/en_ewt-ud-test.conllu', model, tokenizer, 'TF', device)
    #torch.save(data, 'TF_data_sample.pt')

    data = torch.load('TF_data.pt')

    print('begin training')

    train(data)

    


