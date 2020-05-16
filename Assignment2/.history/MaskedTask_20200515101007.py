import os
import torch
import torch.nn as nn
import numpy as np
from dataloaders import EmbeddingDataset
from torch import Tensor
from transformers import *
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

def Task(modeltask, model, tokenizer, source_embeds, W=None, k=0.3, proj=True):
    mask_embed = model(Tensor([103]).unsqueeze(0).type(torch.long).to(device))[-1][0][0][0]
    total_preds = 0
    loss = 0
    loss_func = torch.nn.CosineSimilarity(dim=0)

    for sentence in tqdm(source_embeds):
        if proj:
            proj_t_embeds = torch.matmul(sentence, W).to(device)
        else:
            proj_t_embeds = sentence.to(device)

        proj_t_embeds_masked = proj_t_embeds.clone()
        num_masks = int(sentence.size(0)*k)
        mask_indices = torch.randint(0, sentence.size(0), (num_masks,)).to(device)
        proj_t_embeds_masked[mask_indices] = mask_embed

        pred = modeltask(inputs_embeds=proj_t_embeds_masked.unsqueeze(0))

        predicted_tokens = [torch.argmax(pred[0][0][i]) for i in mask_indices]

        for i, token in enumerate(predicted_tokens):
            pred_embed = model(Tensor([token]).unsqueeze(0).type(torch.long).to(device))[-1][0][0][0]
            original_embed = proj_t_embeds[mask_indices[i]]

            loss_temp = loss_func(pred_embed, original_embed).item()
            loss += loss_temp
            total_preds += 1
        
    score = loss/total_preds
    print('Total Mask Loss: ' + str(score))

    return score

if __name__ == "__main__":
    language = {'en':'bert-base-uncased', 'nl':'bert-base-dutch-cased'}
    model = BertModel.from_pretrained(language['en'], output_hidden_states=True).to(device=device)
    modeltask = BertForMaskedLM.from_pretrained(language['en']).to(device)
    tokenizer = BertTokenizer.from_pretrained(language['en'])
    #W = torch.load('proj_matrix.pt')
    data = torch.load('xlingual_data.pt')
    source_embeds = data['test'][0]

    print('begin task')
    #test for projections
    lambda_c_list = [0.0, 1.0]
    k_list = [0.2, 0.4, 0.6, 0.8]
    for lambda_c in lambda_c_list:
        W = torch.load('proj_matrix_' + str(lambda_c) + '.pt')
        score = Task(modeltask, model, tokenizer, source_embeds, W, k=0.3, proj=True)
    #test with Bertje model
    #Task(modeltask, model, tokenizer, source_embeds, k=0.3, proj=False)


            