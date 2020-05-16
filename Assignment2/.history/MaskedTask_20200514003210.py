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

def Task(modeltask, model, tokenizer, embeddings, W, k=0.3):
    source_embeds = embeddings[0]
    mask_embed = model(Tensor([103]).unsqueeze(0).type(torch.long).to(device))[-1][0][0][0]
    total_preds = 0
    loss = 0
    loss_func = torch.nn.CosineSimilarity(dim=0)

    for sentence in tqdm(source_embeds):
        proj_t_embeds = torch.matmul(sentence, W).to(device)
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
        
    print('Total Mask Loss: ' + str(loss/total_preds))

if __name__ == "__main__":
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(device=device)
    modeltask = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    W = torch.load('proj_matrix.pt')
    data = torch.load('xlingual_data_sample.pt')
    embeddings = data['train']

    print('begin task')
    Task(modeltask, model, tokenizer, embeddings, W, k=0.3)


            