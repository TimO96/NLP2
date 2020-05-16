import torch
import random
from torch import Tensor
from tqdm import tqdm
from typing import List
from conllu import parse_incr, TokenList
from torch.nn.utils.rnn import pad_sequence

# 
def fetch_sen_reps(ud_parses: List[TokenList], model, tokenizer, model_type, concat, device) -> Tensor:
    '''
    creates sentence representations for Transformer or RNN models
    Returns a list with Tensors of shape (num_tokens_in_sentence, representation_size) if concat = False
    Returns a tensor of shape (num_tokens_in_corpus, representation_size) if concat = True
    Note: This is tested on GPT-2, XLNet and LSTM models, other model may have different inputs and outputs
    '''
    sen_reps = []
    sen_len = []
    for sentence in tqdm(ud_parses):
        if model_type=='TF':
            total_tokens = []
            connected = []
            token_num = 0
            for word in sentence:
                input_ids = tokenizer.encode(word['form'])
                total_tokens.extend(input_ids)
                token_num+=len(input_ids)
                connected.append(token_num)
                    
            input_sen = Tensor(total_tokens).type(torch.long).unsqueeze(0).to(device)
            output = model(input_sen)[0][0].detach()
            
            output_sen = output[0:connected[0]].mean(dim=0).unsqueeze(dim=0)
            for i in range(len(connected)-1):
                part = output[connected[i]:connected[i+1]].mean(dim=0).unsqueeze(dim=0)
                output_sen = torch.cat([output_sen, part], dim=0)
                
            if concat:
                sen_reps.extend(output_sen.cpu())
                
            else:
                sen_reps.append(output_sen.cpu())
                
            sen_len.append(len(sentence))
                
        elif model_type=='RNN':
            hidden_0 = model.init_hidden(1)
            input_ids = Tensor([tokenizer['<eos>']]).type(torch.long).unsqueeze(0).to(device)
            _, hidden_eos = model(input_ids, hidden_0)
            output_sen = []
            for word in sentence:
                if word['form'] not in tokenizer:
                    input_ids = Tensor([tokenizer['<unk>']]).type(torch.long).unsqueeze(0).to(device)
                else:
                    input_ids = Tensor([tokenizer[word['form']]]).type(torch.long).unsqueeze(0).to(device)
                  
                output, (hidden, _) = model(input_ids, hidden_eos)
                
                output_sen.append(hidden[-1].squeeze().detach().cpu())

            output_sen = torch.stack(output_sen)
                
            if concat:
                sen_reps.extend(output_sen)
                
            else:
                sen_reps.append(output_sen)
                
            sen_len.append(len(sentence))
    
    if concat:
        return torch.stack(sen_reps)

    else:
        return sen_reps, Tensor(sen_len)

def fetch_pos_tags(ud_parses: List[TokenList], vocab=None) -> Tensor:
    '''
    fetch pos tags for all words in corpus and also returns the mapping POS -> index
    '''
    pos_tags = []
    vocab_list = []
    for sentence in ud_parses:
        for word in sentence:
            pos_tag = word['upostag']
            
            if vocab:
                if pos_tag not in vocab:
                    pos_tags.append(vocab['unk'])
                
                else:
                    pos_tags.append(vocab[pos_tag])
            
            else:
                if pos_tag not in vocab_list:
                    vocab_list.append(pos_tag)
                
                pos_tags.append(vocab_list.index(pos_tag))
    
    if not vocab:
        vocab_list.append('unk')
        vocab_list = {vocab_list[i]:i for i in range(len(vocab_list))}
    
    return Tensor(pos_tags), vocab_list

def fetch_fake_pos_tags(ud_parses, real_vocab, fake_vocab=None):
    '''
    creates fake POS-tags as described in the method of (Hewitt and Liang, 2019)
    '''
    pos_tags = []
    if fake_vocab:
        vocab = fake_vocab
    else:
        vocab = {}

    for sentence in ud_parses:
        for word in sentence:
            if word['form'] not in vocab:
                vocab[word['form']] = random.randint(0, len(real_vocab)-1)
            
            pos_tags.append(vocab[word['form']])
    
    return Tensor(pos_tags), vocab

