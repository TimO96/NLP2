import torch
import random
import string
import pickle
from torch import Tensor
from tqdm import tqdm
from nltk import word_tokenize
from transformers import *
from polyglot.mapping import Embedding
from polyglot.downloader import downloader

def proc_embeddings(data, model, tokenizer, embeddings, nl_en, model_type, device, concat=False):
    '''
    Create embeddings for the projection and english model
    We filter out words that are not in the dictionary or in the dutch embeddings
    Returns: concat=True -> a stack tensor of word embeddings for projection and english resp. 
             concat=False -> a list of sentences with word embeddings for projection and english resp.
    '''
    
    sen_reps_source = []
    sen_reps_target = []
    sen_len = []
    for sentence in tqdm(data):
        output_source = []
        if model_type=='TF':
            total_tokens = [101]

            for word in sentence:
                word_text = word['text'].lower()

                if word_text in embeddings.vocabulary:
                    nl_embed = Tensor(embeddings.get(word_text)).to(device)
                else:
                    continue
                    
                if word_text in nl_en:
                    en_word = nl_en[word_text]
                    input_ids = tokenizer.encode(en_word)[1]
                elif word_text in string.punctuation:
                    input_ids = tokenizer.encode(word_text)[1]
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
        return sen_reps_source, sen_reps_target

def proc_embeddings_bertje(data, model, tokenizer, embeddings, nl_en, model_type, device, concat=False):
    '''
    Create embeddings for the dutch model
    We filter out words that are not in the dictionary or in the dutch embeddings
    Returns: concat=True -> a stack tensor of word embeddings
             concat=False -> a list of sentences with word embeddings
    '''
    
    sen_reps_target = []
    for sentence in tqdm(data):
        output_source = []
        if model_type=='TF':
            total_tokens = [1]

            for word in sentence:
                word_text = word['text'].lower()

                if word_text in embeddings.vocabulary:
                    pass
                else:
                    continue
                    
                if word_text in nl_en or word_text in string.punctuation:
                    input_ids = tokenizer.encode(word_text)[1]
                else:
                    continue
                
                output_source.append(1)
                total_tokens.append(input_ids)

            if not output_source:
                continue
            
            total_tokens.append(2)

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

def fetch_pos_tags(parsed_corpus, embeddings, nl_en, vocab=None):
    '''
    fetch pos tags for all words in corpus and also returns the mapping POS -> index
    '''
    pos_tags = []
    vocab_list = []
    for i, sentence in enumerate(parsed_corpus):
        for word in sentence:
            pos_tag = word['pos']

            if word['text'].lower() not in embeddings.vocabulary:
                continue

            if word['text'].lower() not in nl_en and word['text'].lower() not in string.punctuation:
                continue
            
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

def fetch_fake_pos_tags(parsed_corpus, real_vocab, embeddings, nl_en, fake_vocab=None):
    '''
    creates fake POS-tags as described in the method of (Hewitt and Liang, 2019)
    '''
    pos_tags = []
    if fake_vocab:
        vocab = fake_vocab
    else:
        vocab = {}

    for sentence in parsed_corpus:
        for word in sentence:
            if word['text'].lower() not in embeddings.vocabulary:
                continue

            if word['text'].lower() not in nl_en and word['text'].lower() not in string.punctuation:
                continue

            if word['text'] not in vocab:
                vocab[word['text']] = random.randint(0, len(real_vocab)-1)
            
            pos_tags.append(vocab[word['text']])
    
    return Tensor(pos_tags), vocab

def filter_corpus(parsed_corpus, embeddings, nl_en):
    '''
    Create a corpus of only words with discarded words in embeddings filtered out
    Used in the masking task for the test dataset
    '''
    corpus = []
    for sentence in parsed_corpus:
        sen_words = []
        for word in sentence:
            word_text = word['text'].lower()

            if word_text not in embeddings.vocabulary:
                continue

            if word_text not in nl_en and word_text not in string.punctuation:
                continue

            sen_words.append(word_text)
        
        if sen_words:
            corpus.append(sen_words)
            
    return corpus

    
