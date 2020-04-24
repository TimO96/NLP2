import torch
from conllu import parse_incr
from tqdm import tqdm
from ete3 import Tree as EteTree
from tree import tokentree_to_ete
from scipy.sparse.csgraph import minimum_spanning_tree
from senreps import fetch_sen_reps, fetch_pos_tags


def parse_corpus(filename):

    data_file = open(filename, encoding="utf-8")

    ud_parses = list(parse_incr(data_file))
    
    return ud_parses

def create_gold_distances(corpus):
    '''
    create gold distances from the corpus for every sentence
    gold distances are stored as 2D Tensors
    '''

    all_distances = []

    for item in tqdm(corpus):
        tokentree = item.to_tree()
        ete_tree = tokentree_to_ete(tokentree)
        
        nodes = ete_tree.search_nodes()
        sen_len = len(nodes)
        distances = torch.zeros((sen_len, sen_len))

        for i in range(sen_len):
            node1 = nodes[i]
            for j in range(i, sen_len):
                node2 = nodes[j]
                distance = node1.get_distance(node2)
                distances[int(node1.name)-1][int(node2.name)-1] = distance
                distances[int(node2.name)-1][int(node1.name)-1] = distance

        all_distances.append(distances)

    return all_distances

def create_mst(distances):
    '''
    create minimum spanning trees from distances
    '''
    distances = torch.triu(distances).detach().cpu().numpy()
    
    mst = minimum_spanning_tree(distances).toarray()
    mst[mst>0] = 1.
    
    return mst

def edges(mst):
    '''
    creates edges stored as tuples from minimum spanning trees
    '''
    edges = []
    for i in range(len(mst)):
        for j in (mst[i]==1).nonzero()[0]:
            edges.extend([(i, j), (j, i)])

    return set(edges)

def calc_uuas(pred_distances, gold_distances): 
    '''
    calculating uuas scores for normal struc. dependency task
    '''
    pred_mst = create_mst(pred_distances)
    gold_mst = create_mst(gold_distances)
    
    pred_edges = edges(pred_mst)
    gold_edges = edges(gold_mst)
    
    correct = sum(edge in pred_edges for edge in gold_edges)
    total_gold = len(gold_edges)
    if total_gold!=0:
        uuas = correct/total_gold
    elif len(pred_edges)==0:
        uuas = 1
    else:
        uuas=0
    
    return uuas

def calc_fake_uuas(pred_edges, fake_edges):
    '''
    calculating uuas scores based only on edges for control task
    '''
    correct = sum(pred_edges.max(dim=0)[1]==fake_edges).item()
    if len(fake_edges)!=0:
        uuas = correct/len(fake_edges)
    elif len(pred_edges)==0:
        uuas = 1
    else:
        uuas=0

    return uuas 

def create_data(filename: str, model, tokenizer, model_type, device, vocab=None):
    '''
    create data for the POS-tagging task
    '''
    ud_parses = parse_corpus(filename)
    
    sen_reps = fetch_sen_reps(ud_parses, model, tokenizer, model_type, concat=True, device=device)
    
    pos_tags, pos_vocab = fetch_pos_tags(ud_parses, vocab=vocab)
    
    return sen_reps, pos_tags, pos_vocab

def init_corpus(path, model, tokenizer, model_type, device, concat=False, cutoff=None):
    """ Initialises the data of a corpus.
    
    Parameters
    ----------
    path : str
        Path to corpus location
    concat : bool, optional
        Optional toggle to concatenate all the tensors
        returned by `fetch_sen_reps`.
    cutoff : int, optional
        Optional integer to "cutoff" the data in the corpus.
        This allows only a subset to be used, alleviating 
        memory usage.
    """
    corpus = parse_corpus(path)[:cutoff]
    
    
    embs, sen_len = fetch_sen_reps(corpus, model, tokenizer, model_type, concat=concat, device=device)    
    #gold_distances = create_gold_distances(corpus)
    
    return embs, sen_len