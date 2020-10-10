import spacy
import pickle
import sys
from tqdm import tqdm

#Python 2.7 required for spaCy dutch

def parse_corpus(folder_map='data/'):
    '''
    Parsing train.txt, valid.txt and test.txt through spaCy
    Dumps parsed_corpus for every file in folder
    '''
    nlp = spacy.load("nl_core_news_sm")
    files = ['train', 'valid', 'test']
    for text in files:
        parsed_corpus = []
        with open(folder_map+text+'.txt', 'r') as data: 
            for i, sentence in enumerate(tqdm(data)):
                parsed_corpus.append([])
                proc_sentence = nlp(sentence.decode('utf-8'))
                for token in proc_sentence[:-1]:
                    token_parsed = {}
                    token_parsed['text'] = token.text
                    token_parsed['lemma'] = token.lemma_
                    token_parsed['pos'] = token.pos_
                    token_parsed['tag'] = token.tag_
                    token_parsed['dep'] = token.dep_
                    token_parsed['shape'] = token.shape_
                    token_parsed['is_alpha'] = token.is_alpha
                    token_parsed['is_stop'] = token.is_stop
                    parsed_corpus[i].append(token_parsed)
        
        pickle.dump(parsed_corpus, open("data/parsed_"+text+".pt", "wb"))

