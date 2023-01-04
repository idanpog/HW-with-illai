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
from torch.nn.functional import one_hot
from collections import defaultdict

TRAIN_PATH = "Files/data/train.tagged"
DEV_PATH = "Files/data/dev.tagged"
TRAIN_PATH_WITH_EOF = "train_alt.tagged"
DEV_PATH_WITH_EOF = "dev_alt.tagged"
EOF_EMBED = "my_files/EOF_embed.npy"
OOV_DICT_PATH = "oov_dict.tar"
EMBED_SIZE = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(32)


def split_tags():
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', header=None, names=['Word', 'Tag'])
    return train_df


def update_embeddings(pretrained_model, dataset):
    model = Word2Vec(size=100, min_count=1)
    model.build_vocab(dataset)
    total_examples = model.corpus_count

    # Save the vocab of your dataset
    vocab = list(model.wv.vocab.keys())


def create_oov_dict(train_df, pretrained_word2vec, oov_dict={}):
    for word in train_df['Word']:
        if word == "end_of_sentence":
            oov_dict[word] = np.zeros(EMBED_SIZE)
        if word not in pretrained_word2vec and word not in oov_dict:
            # nearest = pretrained_word2vec.most_similar(word, topn=1)
            oov_dict[word] = np.random.normal(size=EMBED_SIZE)
            # oov_dict[word] = pretrained_word2vec[nearest[0][0]]
    return oov_dict


def create_alt_dev(train=True):
    count = 0
    path = TRAIN_PATH if train else DEV_PATH
    write_path = TRAIN_PATH_WITH_EOF if train else DEV_PATH_WITH_EOF
    with open(path, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == "\n" or line == "\t\n":
                lines[i] = 'End_Of_Sentence\tO\n'
        with open(write_path, 'r+', encoding='utf-8') as w:
            w.seek(0)
            for line in lines:
                w.write(line)
        w.close()
    f.close()




def create_total_oov_dict():
    pretrained_glove = gensim.downloader.load(f'glove-twitter-{EMBED_SIZE}')
    oov_dict = {}
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', header=None, names=['Word', 'Tag'],
                           quoting=csv.QUOTE_NONE)
    test_df = pd.read_csv(DEV_PATH, sep='\t', header=None, names=['Word', 'Tag'],
                           quoting=csv.QUOTE_NONE)
    train_df['Word'] = train_df['Word'].apply(lambda x: str(x).lower())
    test_df['Word'] = test_df['Word'].apply(lambda x: str(x).lower())

    for word in train_df['Word']:
        if word not in pretrained_glove:
            res = bool(re.match("^[A-Za-z0-9_-]*$", word))
            oov_dict[word] = np.random.normal(size=EMBED_SIZE) if res else np.zeros(EMBED_SIZE)
    for word in test_df['Word']:
        if word not in pretrained_glove and word not in oov_dict:
            res = bool(re.match("^[A-Za-z0-9_-]*$", word))
            oov_dict[word] = np.random.normal(size=EMBED_SIZE) if res else np.zeros(EMBED_SIZE)
    torch.save(oov_dict, OOV_DICT_PATH)


def create_embeddings(train=True):
    # Load pretrained embeddings
    pretrained_word2vec = gensim.downloader.load(f'glove-twitter-{EMBED_SIZE}')
    # Load data
    if train:
        train_df = pd.read_csv(TRAIN_PATH_WITH_EOF, sep='\t', header=None, names=['Word', 'Tag'],
                               quoting=csv.QUOTE_NONE)
    else:
        train_df = pd.read_csv(DEV_PATH_WITH_EOF, sep='\t', header=None, names=['Word', 'Tag'], quoting=csv.QUOTE_NONE)
    print(train_df.shape[0])
    # Convert to binary tags
    train_df['Tag'] = train_df['Tag'].apply(lambda row: 0 if row == 'O' else (2 if row == 2 else 1))
    # Convert to lower-case
    train_df['Word'] = train_df['Word'].apply(lambda x: str(x).lower())
    oov_dict = {} if train else torch.load(OOV_DICT_PATH)
    oov_dict = create_oov_dict(train_df, pretrained_word2vec, oov_dict)
    if train:
        torch.save(oov_dict, OOV_DICT_PATH)
    train_embeds = train_df.Word.apply(lambda row: pretrained_word2vec[row] if row in
                                                                               pretrained_word2vec else oov_dict[
        row]).to_frame()
    train_embeds = np.array(train_embeds.Word.to_list())
    tags_arr = np.array(train_df['Tag'].tolist())
    # Count number of positive training samples
    pos_num = train_df['Tag'].sum()
    # Save train embeddings, train labels and positive sample ratio
    if train:
        with open('my_files/train_embeddings.npy', 'wb') as f:
            np.save(f, train_embeds)
            np.save(f, tags_arr)
            np.save(f, np.array(pos_num / train_embeds.shape[0]))
        with open('my_files/EOS_embed.npy', 'wb') as w:
            np.save(w, np.zeros(EMBED_SIZE))
    else:
        with open('my_files/test_embeddings.npy', 'wb') as f:
            np.save(f, train_embeds)
            np.save(f, tags_arr)
            np.save(f, np.array(pos_num / train_embeds.shape[0]))


def create_context_embeddings(train=True):
    addon = "train" if train else "test"
    with open(f'my_files/{addon}_embeddings.npy', 'rb') as f:
        word_embeds = np.load(f, allow_pickle=True).reshape((-1, EMBED_SIZE))
        train_tags = np.load(f)
        pos_ratio = np.load(f)
    f.close()
    with open('my_files/EOS_embed.npy', 'rb') as w:
        EOS_embed = np.load(w)
    # word_embeds = np.extract((word_embeds != EOS_embed), word_embeds).reshape((-1, EMBED_SIZE))
    context_embeds = np.zeros((word_embeds.shape[0], 3 * EMBED_SIZE))
    context_embeds[0] = np.concatenate((EOS_embed, word_embeds[0], word_embeds[1]))
    context_embeds[-1] = np.concatenate((word_embeds[-2], word_embeds[-1], EOS_embed))
    for i in range(1, word_embeds.shape[0] - 1):
        context_embeds[i] = np.concatenate((word_embeds[i - 1], word_embeds[i], word_embeds[i + 1]))
    with open(f'my_files/{addon}_embeddings_with_context.npy', 'wb') as f:
        np.save(f, context_embeds)
        np.save(f, train_tags)
        np.save(f, np.array(pos_ratio))


def add_EOS_tags():
    with open(TRAIN_PATH_WITH_EOF, 'r+') as f:  # r+ does the work of rw
        lines = f.readlines()
        f.seek(0)
        for i, line in enumerate(lines):
            if line == 'End_Of_Sentence\n':
                lines[i] = "End_Of_Sentence" + "\t" + "-1\n"
        for line in lines:
            f.write(line)
    f.close()


# split dev_alt.tagged into sentences
def create_sentence_list(train=True, comp=False):
    path = TRAIN_PATH_WITH_EOF if train else DEV_PATH_WITH_EOF
    df = pd.read_csv(path, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    if comp:
        path = "Files/data/test.untagged"
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
        torch.save(batches, 'my_files/comp_batches.pt')
        torch.save(batch_label_dict, 'my_files/comp_batch_labels.pt')
    else:
        torch.save(batches, f'my_files/train_batches.pkl' if train else f'my_files/test_batches.pkl')
        torch.save(batch_label_dict, f'my_files/train_batch_labels.pkl' if train else f'my_files/test_batch_labels.pkl')

def show_compare_graph(title, train_values, valid_values):
    """
    plot the losses
    """
    plt.figure(f"{title} as a function of epochs", figsize = (8,8))
    plt.plot(range(1, len(train_values)+1), train_values, label="train")
    plt.plot(range(1, len(valid_values)+1), valid_values, label="validation")
    plt.title(title)
    plt.legend()
    plt.savefig(f"{title}_graph.png")

def create_label_dict(tags):
    """
    :param tags: Tags from training data
    :return: Dict for translating tags to integer labels
    """
    d = {}
    idx = 0
    for tag in tags:
        if tag not in d:
            d[tag] = idx
            idx += 1
    return d


def train_SVM():
    start = time()
    # Load pretrained embeddings
    pretrained_glove = gensim.downloader.load(f'glove-twitter-{EMBED_SIZE}')

    # Load data
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', header=None, names=['Word', 'Tag'])
    # Convert to binary tags
    train_df['Tag'] = train_df['Tag'].apply(lambda row: 0 if row == 'O' else 1)
    train_df["Word"].apply(lambda x: str(x).lower())
    # words = train_df['Word'].tolist()
    # for i in range(len(words) - 1):
    #     if words[i + 1]:
    #         words[i] = words[i].capitalize()
    min_count = train_df['Tag'].value_counts().min()
    # Filter data by occurrence frequency
    # filtered_df = train_df.groupby('Tag').head(3 * min_count)
    filtered_df = train_df.groupby('Tag').head(min_count)
    # Split words and tags
    train_words, train_tags = filtered_df['Word'], filtered_df['Tag']
    # Create embeddings for training data
    train_word_embeddings = train_words.apply(lambda row: pretrained_glove[row] if row
                                                                                   in pretrained_glove else np.zeros(
        300)).to_frame()
    numpy_embeddings = np.array(train_word_embeddings.Word.tolist())
    tags_arr = np.array(train_tags.tolist())
    with open('train_embeddings.npy', 'wb') as f:
        np.save(f, numpy_embeddings)
        np.save(f, tags_arr)
    end = time()
    print(f"Embedding calc time: {round(end - start, 4)}")
    start = time()
    # Initialize SVM model
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(numpy_embeddings, tags_arr)
    end = time()
    print(f"SVM fit time: {round(end - start, 4)}")
    # Save trained model
    filename = 'trained_SVM.sav'
    pickle.dump(clf, open(filename, 'wb'))


def predict_SVM():
    """
    :return: Predictions of test word labels
    """
    # Load pretrained word2vec embeddings
    pretrained_word2vec = gensim.downloader.load('word2vec-google-news-300')
    # Load trained SVM model
    loaded_model = pickle.load(open("trained_SVM.sav", 'rb'))
    # Load test data
    test_df = pd.read_csv(DEV_PATH, sep='\t', header=None, names=['Word', 'Tag'])
    # Create binary test labels
    test_df['Tag'] = test_df['Tag'].apply(lambda row: 0 if row == 'O' else 1)
    # Split words and tags
    test_words, test_tags = test_df['Word'], test_df['Tag']
    # Create word embeddings
    test_word_embeddings = test_words.apply(lambda row: pretrained_word2vec[row] if row
                                                                                    in pretrained_word2vec else np.zeros(
        300)).to_frame()
    X_test = np.array(test_word_embeddings.Word.tolist())
    Y_test = np.array(test_tags.tolist())

    preds = loaded_model.predict(X_test)
    f1 = sklearn.metrics.f1_score(Y_test, preds)
    print(f1)


class NeuralNet(nn.Module):
    def __init__(self, input_size=3 * EMBED_SIZE, hidden_size=512, output_size=2):
        super().__init__()
        self.input_size = input_size
        self.hidden1_size = hidden_size * 4
        self.hidden2_size = hidden_size * 2
        self.hidden3_size = hidden_size
        self.hidden4_size = hidden_size
        self.hidden5_size = hidden_size // 2
        self.output_size = output_size

        self.layer1 = nn.Linear(self.input_size, self.hidden1_size)
        self.layer2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.layer3 = nn.Linear(self.hidden2_size, self.hidden3_size)
        self.layer4 = nn.Linear(self.hidden3_size, self.hidden4_size)
        self.layer5 = nn.Linear(self.hidden4_size, self.hidden5_size)
        self.layer6 = nn.Linear(self.hidden5_size, self.output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        z = self.relu(self.layer1(x))
        z = self.dropout(z)
        z = self.relu(self.layer2(z))
        # z = self.dropout(z)
        z = self.relu(self.layer3(z))
        # z = self.dropout(z)
        z = self.relu(self.layer4(z))
        z = self.relu(self.layer5(z))
        z = self.layer6(z)
        return self.softmax(z).squeeze()


class FC(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, output_size=2):
        super().__init__()
        self.input_size = input_size
        self.hidden1_size = hidden_size
        self.hidden2_size = hidden_size // 2
        # self.hidden3_size = hidden_size // 2
        self.output_size = output_size
        self.dropout = nn.Dropout(0.3)

        self.layer1 = nn.Linear(self.input_size, self.hidden1_size)
        self.layer2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.layer3 = nn.Linear(self.hidden2_size, self.output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        z = self.relu(self.layer1(x))
        z= self.dropout(z)
        z = self.relu(self.layer2(z))
        z = self.dropout(z)
        z = self.layer3(z)
        return z


class BatchedSenteceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # dict of batch tensors
        self.labels = labels  # dict of batch labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CustomEmbsDataset(Dataset):
    def __init__(self, path='my_file/train_embeddings_with_context.npy', transform=None, target_transform=None,
                 train=True):
        with open(path, 'rb') as f:
            # self.word_embeds = torch.tensor(np.load(f)).to(device=device, dtype=torch.float)
            # self.train_tags = torch.tensor(np.load(f)).to(device=device, dtype=torch.float)
            self.train = train
            self.word_embeds = torch.tensor(np.load(f))
            self.train_tags = np.load(f)
            self.pos_ratio = np.load(f)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.word_embeds.shape[0]

    def __getitem__(self, idx):
        word = self.word_embeds[idx]
        label = self.train_tags[idx]
        return word, label


# LSTM model that recieves word embeddings and outputs a probability of being a named entity
class old_LSTM(nn.Module):
    def __init__(self, input_size=EMBED_SIZE, hidden_size=EMBED_SIZE, output_size=2, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True)
        self.fc = FC(input_size=self.hidden_size, hidden_size=EMBED_SIZE, output_size=self.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.dropout(out)

        # Decode the hidden state of each time step
        out = self.fc(out[:, :, :]).squeeze(0)
        return self.sigmoid(out)



class CompetitiveModel(nn.Module):
    def __init__(self, lstm, fc):
        super().__init__()
        self.LSTM = lstm
        self.FC = fc

    def forward(self, x):
        x = self.LSTM(x)
        x = self.FC(x)
        return x


def train_FF():
    num_epochs = 2000
    lr = 0.00008

    batch_size = 256
    # Load dataset and create dataloader
    train_set = CustomEmbsDataset("my_files/train_embeddings_with_context.npy")
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,
                                  persistent_workers=True, pin_memory=True)
    test_set = CustomEmbsDataset("my_files/test_embeddings_with_context.npy")
    test_loader = DataLoader(test_set, batch_size=2 * batch_size, shuffle=False, num_workers=4,
                             persistent_workers=True, pin_memory=True)
    model = NeuralNet().to(device)
    class_weights = torch.tensor([float(train_set.pos_ratio), 1 - float(train_set.pos_ratio)], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(lr=lr, params=model.parameters())
    lam = lambda epoch: 0.95 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
    max_f1 = 0
    for t in (tq := tqdm(range(num_epochs), position=0, leave=True)):
        # indices = torch.randperm(word_embeds.size()[0])
        tp, tn, fp, fn, accuracy = 0, 0, 0, 0, 0

        # optimizer.zero_grad()
        for i, (words, labels) in enumerate(train_dataloader):
            model.train()

            optimizer.zero_grad()
            words, labels = words.to(device, torch.float), labels.to(device, torch.long)

            outputs = model(words)

            loss = criterion(outputs, labels)
            loss.backward()

            # torch.nn.utils.clip_grad_norm(model.parameters(), 1)

            optimizer.step()
            scheduler.step()

        f1, precision, recall = calc_test_f1(test_loader, model)

        # train_f1, train_precision, train_recall = calc_test_f1(train_dataloader, model)
        # tq.set_description(f"")
        if f1 > max_f1:
            torch.save(model.state_dict(), f"best_new_FF_model.pkl")
            max_f1 = f1
        tq.set_description(f"Loss: {loss.item()}, F1 score: {f1}, Precision: {precision}, Recall: {recall}")
        tq.set_postfix_str(f"Max F1: {max_f1}")


def calc_test_f1(test_loader, model):
    model.eval()
    with torch.no_grad():
        tp, tn, fp, fn = 0, 0, 0, 0
        for i, (words, labels) in enumerate(test_loader):
            words, labels = words.to(device, torch.float), labels.to(device, torch.long).squeeze()
            outputs = model(words)
            preds = torch.argmax(outputs, dim=1)
            tp += ((preds == labels) * (preds == 1)).sum()
            tn += ((preds == labels) * (preds == 0)).sum()
            fp += ((preds != labels) * (preds == 1)).sum()
            fn += ((preds != labels) * (preds == 0)).sum()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    return round(float(f1), 4), round(float(precision), 4), round(float(recall), 4)


def sum_model_params():
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load("FF_model.pkl"))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def eval_best_model():
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load("best_FF_model.pkl"))
    test_set = CustomEmbsDataset("my_files/test_embeddings_with_context.npy")
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4,
                             persistent_workers=True, pin_memory=True)
    f1, precision, recall = calc_test_f1(test_loader, model)
    print(f1, precision, recall)

# LSTM model for named entity recognition
class MyLSTM(nn.Module):
    def __init__(self, input_size=EMBED_SIZE, hidden_size=EMBED_SIZE, num_layers=2, num_classes=2):
        super(MyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc1 = nn.Linear(2*hidden_size, num_classes)
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
        # out = self.relu(out)
        # out = self.relu(self.fc2(out))
        return self.softmax(out)


def count_train_pos():
    with open(TRAIN_PATH, "rb") as f:
        lines = f.readlines()
        pos = 0
        for i, line in enumerate(lines):
            if not line.endswith(b"O\n"):
                pos += 1
    return pos / len(lines)


def train_competitive():
    num_epochs = 2000
    lr = 0.0005

    batch_size = 1
    r_batch_size = 1
    # Load dataset and create dataloader
    train_data = torch.load("my_files/train_batches.pkl")
    train_tags = torch.load("my_files/train_batch_labels.pkl")
    train_set = BatchedSenteceDataset(train_data, train_tags)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                                  persistent_workers=True, pin_memory=True)
    test_data = torch.load("my_files/test_batches.pkl")
    test_tags = torch.load("my_files/test_batch_labels.pkl")
    test_set = BatchedSenteceDataset(test_data, test_tags)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2,
                             persistent_workers=True, pin_memory=True)
    model = MyLSTM().to(device)
    pos_ratio = count_train_pos()
    class_weights = torch.tensor([float(pos_ratio), 1 - float(pos_ratio)], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(lr=lr, params=model.parameters())
    # lam = lambda epoch: 0.8 ** epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
    max_f1 = 0
    # train loop
    for t in (tq := tqdm(range(num_epochs), position=0, leave=True)):
        for i, (words, tags) in enumerate(train_dataloader):
            model.train()

            # optimizer.zero_grad()
            tags = tags.to(device, torch.long).squeeze(0)
            words = words.to(device, torch.float)
            if i % r_batch_size == 0 or len(train_dataloader) - i < r_batch_size:
                # optimizer.zero_grad()
                labels = tags
                outputs = model(words)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()
            else:
                outputs = torch.cat((outputs, model(words)), dim=0)
                labels = torch.cat((labels, tags), dim=0)

            torch.nn.utils.clip_grad_norm(model.parameters(), 1)

            # optimizer.step()

        f1, precision, recall = calc_test_f1(test_loader, model)

        if f1 > max_f1:
            torch.save(model.state_dict(), "best_model.pkl")
            max_f1 = f1
        # train_f1, train_precision, train_recall = calc_test_f1(train_dataloader, model)
        # tq.set_description(f"")
        tq.set_description(f"Loss: {loss.item()}, F1 score: {f1}, Precision: {precision}, Recall: {recall}")
        tq.set_postfix_str(f"Max F1: {max_f1}")

def predict_comp():
    model = MyLSTM().to(device)
    model.load_state_dict(torch.load("best_model.pkl"))
    test_set = BatchedSenteceDataset(torch.load("comp_batches.pkl"), torch.load("comp_batch_labels.pkl"))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2,
                             persistent_workers=True, pin_memory=True)
    model.eval()
    with torch.no_grad():
        tp, tn, fp, fn = 0, 0, 0, 0
        preds = None
        for i, (words, labels) in enumerate(test_loader):
            words, labels = words.to(device, torch.float), labels.to(device, torch.long).squeeze()
            outputs = model(words)
            preds = torch.argmax(outputs, dim=1) if preds is None else torch.cat((preds, torch.argmax(outputs, dim=1)), dim=0)
            #preds = torch.cat(preds, torch.argmax(outputs, dim=1))
            # tp += ((preds == labels) * (preds == 1)).sum()
            # tn += ((preds == labels) * (preds == 0)).sum()
            # fp += ((preds != labels) * (preds == 1)).sum()
            # fn += ((preds != labels) * (preds == 0)).sum()
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # f1 = 2 * (precision * recall) / (precision + recall)
    # f1, precision, recall = calc_test_f1(test_loader, model)
    # print(f1, precision, recall)
    return preds

def save_predict_preds():
    create_sentence_batches(train=False, comp=True)
    preds = predict_comp()
    lines = ""
    with open("Files/data/test.untagged", "r+", encoding="utf-8") as f:
        lines = f.readlines()
        new_lines = []
        i = 0
        for line in lines:
            if not line.strip():
                new_lines+=["\n"]
                continue
            new_lines += line + "\t" + str(preds[i].item()) + "\n"
            i+=1
    with open("Files/data/test.untagged", "w+") as w:
        for line in new_lines:
            w.write(line)





if __name__ == '__main__':
    # train_SVM()
    # create_alt_dev(train=True)
    # create_alt_dev(train=False)
    # create_embeddings(train=True)
    # print("Train Embeddings Done")
    # create_context_embeddings(train=True)
    # print("Train Context Embeddings Done")
    # create_embeddings(train=False)
    # print("Test Embeddings Done")
    # create_context_embeddings(train=False)
    # print("Test Context Embeddings Done")
    # eval_best_model()
    # train_FF()
    # create_total_oov_dict()
    # print("Done")
    # create_sentence_batches(train=True)
    # print("Train batches done")
    # create_sentence_batches(train=False)
    # print("Test batches done")
    # train_competitive()
    save_predict_preds()