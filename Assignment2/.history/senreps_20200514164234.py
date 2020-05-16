import torch
import random
import string
from torch import Tensor
from tqdm import tqdm
from conllu import parse_incr, TokenList

def proc_embeddings(data, model, tokenizer, embeddings, nl_en, model_type, device, concat=False):
    
    sen_reps_source = []
    sen_reps_target = []
    sen_len = []
    for sentence in tqdm(data):
        output_source = []
        if model_type=='TF':
            total_tokens = [101]

            for word in sentence:
                if word in embeddings.vocabulary:
                    nl_embed = Tensor(embeddings.get(word)).to(device)
                else:
                    continue
                    
                if word in nl_en:
                    en_word = nl_en[word]
                    input_ids = tokenizer.encode(en_word)[1]
                elif word in string.punctuation:
                    input_ids = tokenizer.encode(word)[1]
                else:
                    continue
                    
                output_source.append(nl_embed)
                total_tokens.append(input_ids)

            if not output_source:
                continue
            
            total_tokens.append(102)
            output_source = torch.stack(output_source)

            input_sen = Tensor(total_tokens).type(torch.long).unsqueeze(0).to(device)
            output_sen = model(input_sen)[-1][0][0][1:-1].detach()

            if concat:
                sen_reps_target.extend(output_sen.cpu())
                sen_reps_source.extend(output_source.cpu())
                

            else:
                sen_reps_target.append(output_sen.cpu())
                sen_reps_source.append(output_source.cpu())

            sen_len.append(output_sen.size(0))

    if concat:
        return torch.stack(sen_reps_source), torch.stack(sen_reps_target)

    else:
        return sen_reps_source, sen_reps_target, Tensor(sen_len)

def proc_embeddings_bertje(data, model, tokenizer, embeddings, nl_en, model_type, device, concat=False):
    
    sen_reps_target = []
    for sentence in tqdm(data):
        output_source = []
        if model_type=='TF':
            total_tokens = [101]

            for word in sentence:
                if word in embeddings.vocabulary:
                    pass
                else:
                    continue
                    
                if word in nl_en:
                    input_ids = tokenizer.encode(word)[1]
                else:
                    continue
                
                output_source.append(1)
                total_tokens.append(input_ids)

            if not output_source:
                continue
            
            total_tokens.append(102)

            input_sen = Tensor(total_tokens).type(torch.long).unsqueeze(0).to(device)
            output_sen = model(input_sen)[-1][0][0][1:-1].detach()

            if concat:
                sen_reps_target.extend(output_sen.cpu())
                
            else:
                sen_reps_target.append(output_sen.cpu())

    if concat:
        return torch.stack(sen_reps_target)

    else:
        return sen_reps_target
