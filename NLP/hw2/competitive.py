import sklearn
import torch
import torch.nn as nn
from sklearn.svm import SVC
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
import gensim.downloader
import lovely_tensors as lt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from time import time
import pickle
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from collections import defaultdict

TRAIN_PATH = "Files/data/train.tagged"
DEV_PATH = "Files/data/dev.tagged"
TRAIN_PATH_WITH_EOF = "my_files/train_alt.tagged"
DEV_PATH_WITH_EOF = "my_files/dev_alt.tagged"
EOF_EMBED = "my_files/EOF_embed.npy"
OOV_DICT_PATH = "my_files/oov_dict.tar"
EMBED_SIZE = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(32)


class TransformerModel(nn.Module):
    def __init__(self, embedding_size, input_size, d_model, output_size):
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size

        dropout = 0.3
        self.encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=6, dim_feedforward=256, dropout=dropout,
                                                   batch_first=True)






class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout3d(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)