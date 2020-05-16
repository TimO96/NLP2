import os
import torch
import torch.nn as nn
import numpy as np
from dataloaders import EmbeddingDataset
from torch import Tensor
from transformers import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

def Task(modeltask, model, tokenizer, embeddings, W, k=0.3):
    source_embeds = embeddings[0]
    target_embeds = embeddings[1]
    mask_embed = model(Tensor([103]).unsqueeze(0).type(torch.long).to(device))

    for sentence in source_embeds:
        proj_t_embeds = torch.matmul(source_embeds, W)
        num_masks = int(sentence.size(0)*k)
        mask_indices = torch.randint(0, sentence.size(0), (num_masks,))
        proj_t_embeds[mask_indices] = 


        pred = model(inputs_embeds=input_embs)