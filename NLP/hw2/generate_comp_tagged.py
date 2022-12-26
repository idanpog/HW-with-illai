import csv
import re
import sklearn
import torch
import torch.nn as nn
from sklearn.svm import SVC
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
import gensim.downloader
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from time import time
import pickle
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

TRAIN_PATH = "Files/data/train.tagged"
DEV_PATH = "Files/data/dev.tagged"
TRAIN_PATH_WITH_EOF = "train_alt.tagged"
DEV_PATH_WITH_EOF = "dev_alt.tagged"
OOV_DICT_PATH = "oov_dict.tar"
EMBED_SIZE = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(32)


def create_sentence_list(train=True, comp=False):
    path = TRAIN_PATH_WITH_EOF if train else DEV_PATH_WITH_EOF
    if comp:
        path = "test.untagged"
    with open(path, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        sentences = []
        sentence = []
        for i, line in enumerate(lines):
            if line == 'End_Of_Sentence\tO\n':
                sentences.append(sentence)
                sentence = []
            else:
                line = line.lower()
                sentence.append(line.split('\t'))
    f.close()
    return sentences


def create_sentence_batches(train=True, comp=False):
    pretrained_glove = gensim.downloader.load(f'glove-twitter-{EMBED_SIZE}')
    batches = {}
    batch_label_dict = {}
    sentences = create_sentence_list(train)
    oov_dict = torch.load(OOV_DICT_PATH)
    embed = lambda word: pretrained_glove[word] if word in pretrained_glove else oov_dict[word]
    tag = lambda tag: 0 if tag == 'o' or tag == 'o\n' else 1
    for i, sentence in enumerate(sentences):
        l = [torch.tensor(embed(word[0])) for word in sentence]
        batch = torch.cat(l, dim=0) \
            .reshape(-1, EMBED_SIZE)
        batch_labels = torch.tensor([int(tag(word[1])) for word in sentence])
        batches[i] = batch
        batch_label_dict[i] = batch_labels
    if comp:
        torch.save(batches, 'comp_batches.pkl')
        torch.save(batch_label_dict, 'comp_batch_labels.pkl')
    else:
        torch.save(batches, f'train_batches.pkl' if train else f'test_batches.pkl')
        torch.save(batch_label_dict, f'train_batch_labels.pkl' if train else f'test_batch_labels.pkl')


class BatchedSenteceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # dict of batch tensors
        self.labels = labels  # dict of batch labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MyLSTM(nn.Module):
    def __init__(self, input_size=EMBED_SIZE, hidden_size=EMBED_SIZE, num_layers=2, num_classes=2):
        super(MyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc1 = nn.Linear(2 * hidden_size, num_classes)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc1(out[:, :, :]).squeeze(0)
        return self.softmax(out)


def predict_comp():
    model = MyLSTM().to(device)
    model.load_state_dict(torch.load("best_model.pkl"))
    test_set = BatchedSenteceDataset(torch.load("comp_batches.pkl"),
                                     torch.load("comp_batch_labels.pkl"))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2,
                             persistent_workers=True, pin_memory=True)
    model.eval()
    with torch.no_grad():
        for i, (words, labels) in enumerate(test_loader):
            words, labels = words.to(device, torch.float), labels.to(device, torch.long).squeeze()
            outputs = model(words)
            preds = torch(preds, torch.argmax(outputs, dim=1))
    return preds


def save_predict_preds():
    create_sentence_batches(train=False, comp=True)
    preds = predict_comp()
    with open("test.untagged", "r+") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = line + "\t" + preds[i]
    with open("test.untagged", "w+") as w:
        for line in lines:
            w.write(line)

if __name__ == "__main__":
    save_predict_preds()
    print("done")