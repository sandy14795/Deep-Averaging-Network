import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])

def load_vectors(fname, vocab ):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in vocab:
            data[tokens[0]] = tokens[1:]
    return data

def load_embedding(vocab, emb_file, emb_size, embd):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """

    data = load_vectors(emb_file, vocab)
    emb = np.copy(embd)
    for word, embd in data.items():
        emb[vocab[word]] = np.array(list(map(float, embd)))

    return emb


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            # dummy = self.embed.weight.data.numpy() 
            # emb_mtx = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size, dummy)
            # with open('embs_cfimdb.npy','wb') as f:
            #     np.save(f,emb_mtx)
            if 'cfimdb' in self.args.train:
                emb_mtx = np.load('embs_cfimdb.npy')
            else:
                emb_mtx = np.load('embs_sst.npy')
            self.copy_embedding_from_numpy(emb_mtx)

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        """
        
        self.embed = nn.Embedding(len(self.vocab) , self.args.emb_size )
        self.fc1 = nn.Linear(self.args.emb_size, self.args.hid_size)
        self.fc2 = nn.Linear(self.args.hid_size, self.args.hid_size)
        self.fc3 = nn.Linear(self.args.hid_size, self.tag_size)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        """
        ## to_do Xavier initialisations
        # if not self.args.emb_file:
        torch.nn.init.xavier_uniform_(self.embed.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def copy_embedding_from_numpy(self, pretrained_weights):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        """
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_weights))
        # self.embed.weight.data.copy_(pretrained_weights)
        self.embed.weight.requires_grad = True
    
    def dropout(self, x):
        for idx, batch_x in enumerate(x):
            probs = torch.empty(len(batch_x)).uniform_(0, 1)
            x[idx] = torch.where(probs >= 0.2, batch_x.data, torch.zeros(len(batch_x)))
        return x
    
            
    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        #to_do Xavier initialisations, dropouts, pretrained embeddings
        
        x = self.embed(x)
        x = x.mean(dim=1)
        # x = self.dropout(x)
        x = self.fc1(x)
        # x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.leaky_relu(self.fc3(x))
        return x
