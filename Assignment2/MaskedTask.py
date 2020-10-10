import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import string
from dataloaders import EmbeddingDataset
from torch import Tensor
from transformers import *
from tqdm import tqdm
from polyglot.mapping import Embedding
from polyglot.downloader import downloader
from collections import Counter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.manual_seed(42)
#np.random.seed(42)

def Task(modeltask, model, tokenizer, source_embeds, corpus, nl_en, en_nl, language='en', W=None, k=1, proj=True):
    '''
    performs the masking task where k controls the total masking of the sentence
    if dutch projections -> proj=True
    otherwise -> proj=False
    Returns tuple of (score and perplexity) and a Counter of word predictions
    '''
    pred_words = Counter()
    total_preds = 0
    loss = 0
    total_pp_loss = 0
    pp_func = torch.nn.CrossEntropyLoss()

    for i, sentence in enumerate(tqdm(source_embeds)):
        #project embeddings to new space if proj=True
        if proj:
            proj_t_embeds = torch.matmul(sentence, W).to(device)
        else:
            proj_t_embeds = sentence.to(device)

        proj_t_embeds_masked = proj_t_embeds.clone()

        #Create sentence tokens that are fed to the model
        if language == 'en':
            sen_tokens = [101]
        elif language == 'nl':
            sen_tokens = [1]

        for word in corpus[i]:
            if language == 'en':
                if word in string.punctuation:
                    token = tokenizer.encode(word)[1]
                else:
                    token = tokenizer.encode(nl_en[word])[1]
            elif language == 'nl':
                token = tokenizer.encode(word)[1]

            sen_tokens.append(token)

        if language == 'en':
            sen_tokens.append(102)
        elif language == 'nl':
            sen_tokens.append(2)

        sen_tokens = Tensor(sen_tokens).type(torch.long).to(device)
        #set the amount of masking
        if isinstance(k, float):
            k = int(np.ceil(sentence.size(0)*k))
        
        mask_indices = torch.randperm(sentence.size(0)).to(device)[:k] + 1
        masked_tokens = sen_tokens[mask_indices]
        if language == 'en':
            sen_tokens[mask_indices] = 103
        elif language == 'nl':
            sen_tokens[mask_indices] = 4

        en_output = model(sen_tokens.unsqueeze(0))[-1][0][0]

        masked_output = en_output[mask_indices]
        proj_t_embeds_masked = torch.cat((en_output[0].unsqueeze(0), proj_t_embeds_masked, en_output[-1].unsqueeze(0)), 0)
        proj_t_embeds_masked[mask_indices] = masked_output

        pred = modeltask(inputs_embeds=proj_t_embeds_masked.unsqueeze(0))

        predicted_tokens = [torch.argmax(pred[0][0][mask]) for mask in mask_indices]

        for j, token in enumerate(predicted_tokens):

            original_token = masked_tokens[j].unsqueeze(0)

            if tokenizer.decode(original_token) in string.punctuation:
                continue
            
            #Use full distribution to calculate the cross entropy loss
            CE_loss = pp_func(pred[0][0][mask_indices[j]].unsqueeze(0), original_token).item()
            loss += CE_loss

            pp_loss = 2**(CE_loss)

            total_pp_loss += pp_loss

            pred_word = tokenizer.decode(token.unsqueeze(0))

            if language == 'en':
                if pred_word not in string.punctuation and pred_word in en_nl:
                    pred_word = en_nl[pred_word]

            pred_words[pred_word] += 1

            total_preds += 1
        
        if i%1000==0:
            print(loss/total_preds, total_pp_loss/total_preds, total_preds)
        
    score = loss/total_preds
    perplexity = total_pp_loss/total_preds
    print('CE Loss: ' + str(score))
    print('Perplexity: ' + str(perplexity))
    print('total_predictions: ' + str(total_preds))
    print(pred_words.most_common(10))

    return (score, perplexity), pred_words


            