import torch
from torch import Tensor
from tqdm import tqdm
from typing import List
from conllu import parse_incr, TokenList
from torch.nn.utils.rnn import pad_sequence

# Should return a tensor of shape (num_tokens_in_corpus, representation_size)
# Make sure you correctly average the subword representations that belong to 1 token!
def fetch_sen_reps(ud_parses: List[TokenList], model, tokenizer, model_type, concat, device) -> Tensor:
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
    
    return sen_reps, Tensor(sen_len)