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
    # path = TRAIN_PATH_WITH_EOF if train else DEV_PATH_WITH_EOF
    path = "test.untagged"
    with open(path, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        sentences = []
        sentence = []
        for i, line in enumerate(lines):
            if line == '\n':
                sentences.append(sentence)
                sentence = []
            else:
                line = line.lower()
                sentence.append(line)
    f.close()
    return sentences


def embed(pretrained_glove, oov_dict, word):
    if word in pretrained_glove:
        return pretrained_glove[word]
    if not bool(re.match("^[A-Za-z0-9_-]*$", word)):
        return torch.zeros(EMBED_SIZE)
    if word in oov_dict:
        return oov_dict[word]
    else:
        return torch.randn(EMBED_SIZE)


def create_sentence_batches(train=True, comp=False):
    pretrained_glove = gensim.downloader.load(f'glove-twitter-{EMBED_SIZE}')
    batches = {}
    sentences = create_sentence_list(train)
    oov_dict = torch.load(OOV_DICT_PATH)
    for i, sentence in enumerate(sentences):
        l = [torch.tensor(embed(pretrained_glove, oov_dict, word)) for word in sentence]
        batch = torch.cat(l, dim=0).reshape(-1, EMBED_SIZE)
        batches[i] = batch
    torch.save(batches, 'comp_batches.pkl')


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
    comp_sentences = torch.load("comp_batches.pkl")
    # test_set = BatchedSenteceDataset(torch.load("comp_batches.pkl"),
    #                                  torch.load("comp_batch_labels.pkl"))
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2,
    #                          persistent_workers=True, pin_memory=True)
    model.eval()
    preds = None
    with torch.no_grad():
        for sentence in enumerate(comp_sentences.values()):
            sentence = sentence[1].unsqueeze(0).to(device)
            outputs = model(sentence)
            preds = torch.argmax(outputs, dim=1) if preds == None else torch.cat((preds, torch.argmax(outputs, dim=1)), dim=0)
    return preds


def save_predict_preds():
    create_sentence_batches(train=False, comp=True)
    preds = predict_comp()
    with open("test.untagged", "r+", encoding='utf-8') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            if line != '\n':
                lines[i] = line.strip() + "\t" + str(preds[i].item()) + "\n"
                i += 1
    with open("test.tagged", "w+", encoding='utf-8') as w:
        for line in lines:
            w.write(line)


if __name__ == "__main__":
    save_predict_preds()
    print("done")