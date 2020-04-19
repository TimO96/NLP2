import torch
from conllu import parse_incr
from tqdm import tqdm
from ete3 import Tree as EteTree
from tree import tokentree_to_ete
from scipy.sparse.csgraph import minimum_spanning_tree
from senreps import fetch_sen_reps


def parse_corpus(filename):

    data_file = open(filename, encoding="utf-8")

    ud_parses = list(parse_incr(data_file))
    
    return ud_parses

def create_gold_distances(corpus):
    all_distances = []

    for item in tqdm(corpus):
        tokentree = item.to_tree()
        ete_tree = tokentree_to_ete(tokentree)
        
        nodes = ete_tree.search_nodes()
        sen_len = len(nodes)
        distances = torch.zeros((sen_len, sen_len))

        # Your code for computing all the distances comes here.
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
    distances = torch.triu(distances).detach().cpu().numpy()
    
    mst = minimum_spanning_tree(distances).toarray()
    mst[mst>0] = 1.
    
    return mst

def edges(mst):
    # Your code for retrieving the edges from the MST matrix
    edges = []
    for i in range(len(mst)):
        for j in (mst[i]==1).nonzero()[0]:
            edges.extend([(i, j), (j, i)])

    return set(edges)

def calc_uuas(pred_distances, gold_distances): 
    # Your code for computing the UUAS score
    pred_mst = create_mst(pred_distances)
    gold_mst = create_mst(gold_distances)
    
    pred_edges = edges(pred_mst)
    gold_edges = edges(gold_mst)
    
    correct = sum(edge in pred_edges for edge in gold_edges)
    total_gold = len(gold_edges)
    if total_gold!=0:
        uuas = correct/total_gold
    elif total_gold==0 and len(pred_edges)==0:
        uuas = 1
    else:
        uuas=0
    
    return uuas

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
    gold_distances = create_gold_distances(corpus)
    
    return gold_distances, embs, sen_len