import torch
import torch.nn as nn
import random
from tqdm import tqdm


class StructuralProbe(nn.Module):
    """ Computes squared L2 distance after projection by a matrix.
    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """

    def __init__(self, model_dim, rank, device):
        super().__init__()
        self.probe_rank = rank
        self.model_dim = model_dim
        
        self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))
        
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, batch):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.
        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        
        batchlen, seqlen, rank = transformed.size()
        
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1,2)
        
        diffs = transformed - transposed
        
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)

        return squared_distances

    
class L1DistanceLoss(nn.Module):
    """Custom L1 loss for distance matrices."""
    def __init__(self):
        super().__init__()

    def forward(self, predictions, label_batch, length_batch):
        """ Computes L1 loss on distance matrices.
        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the square of the sentence length)
        and then across the batch.
        Args:
          predictions: A pytorch batch of predicted distances
          label_batch: A pytorch batch of true distances
          length_batch: A pytorch batch of sentence lengths
        Returns:
          A tuple of:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        total_sents = torch.sum((length_batch != 0)).float()
        squared_lengths = length_batch.pow(2).float()

        if total_sents > 0:
            loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=(1,2))
            normalized_loss_per_sent = loss_per_sent / squared_lengths
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        
        else:
            batch_loss = torch.tensor(0.0)
        
        return batch_loss, total_sents

class CrossEntropyLoss(nn.Module):
    """Custom cross-entropy loss"""
    def __init__(self):
      super(CrossEntropyLoss, self).__init__()
      self.pytorch_ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

    def forward(self, predictions, label_batch, length_batch):
      """
      Computes and returns CrossEntropyLoss.
      Ignores all entries where label_batch=-1
      Noralizes by the number of sentences in the batch.
      Args: 
        predictions: A pytorch batch of logits
        label_batch: A pytorch batch of label indices
        length_batch: A pytorch batch of sentence lengths
      Returns:
        A tuple of:
          cross_entropy_loss: average loss in the batch
          total_sents: number of sentences in the batch
      """
      batchlen, seqlen, class_count = predictions.size()
      total_sents = torch.sum((length_batch != 0)).float()
      predictions = predictions.view(batchlen*seqlen, class_count)
      label_batch = label_batch.view(batchlen*seqlen).long()
      cross_entropy_loss = self.pytorch_ce_loss(predictions, label_batch) / total_sents

      return cross_entropy_loss, total_sents

class TwoWordBilinearLabelProbe(nn.Module):
    """ Computes a bilinear function of pairs of vectors.
    For a batch of sentences, computes all n^2 pairs of scores
    for each sentence in the batch.
    """
    def __init__(self, model_dim, rank, prob, device):
        super(TwoWordBilinearLabelProbe, self).__init__()

        self.maximum_rank = rank
        self.model_dim = model_dim
        self.proj_L = nn.Parameter(data = torch.zeros(self.model_dim, self.maximum_rank))
        self.proj_R = nn.Parameter(data = torch.zeros(self.maximum_rank, self.model_dim))
        self.bias = nn.Parameter(data=torch.zeros(1))
        nn.init.uniform_(self.proj_L, -0.05, 0.05)
        nn.init.uniform_(self.proj_R, -0.05, 0.05)
        nn.init.uniform_(self.bias, -0.05, 0.05)
        self.to(device)
        self.dropout = nn.Dropout(p=prob)

    def forward(self, batch):
        """ Computes all n^2 pairs of attachment scores
        for each sentence in a batch.
        Computes h_i^TAh_j for all i,j
        where A = LR, L in R^{model_dim x maximum_rank}; R in R^{maximum_rank x model_rank}
        hence A is rank-constrained to maximum_rank.
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of scores of shape (batch_size, max_seq_len, max_seq_len)
        """
        batchlen, seqlen, rank = batch.size()
        batch = self.dropout(batch)
        proj = torch.mm(self.proj_L, self.proj_R)
        batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank)
        batch_transposed = batch.unsqueeze(1).expand(batchlen, seqlen, seqlen, rank).contiguous().view(batchlen*seqlen*seqlen,rank,1)

        psd_transformed = torch.matmul(batch_square.contiguous(), proj).view(batchlen*seqlen*seqlen,1, rank)
        logits = (torch.bmm(psd_transformed, batch_transposed) + self.bias).view(batchlen, seqlen, seqlen)
        return logits

def make_fake_labels(ud_parses, fake_vocab=None):
    '''
    make fake labels for the structural dependency task
    Returns single labels for every sentence in a Tensor
    '''

    labels = []
    if fake_vocab:
        vocab = fake_vocab
    else:
        vocab = {}

    for sentence in tqdm(ud_parses):
        label = torch.zeros((len(sentence)))
        for i, word in enumerate(sentence):  
            if word['form'] not in vocab:
                vocab[word['form']] = random.randint(0, 2)
            
            output = vocab[word['form']]

            if output==0:
                label[i] = i
              
            elif output==1:
                label[i] = 0
            
            else:
                label[i] = len(sentence)-1
        
        labels.append(label)
    
    return labels, vocab



