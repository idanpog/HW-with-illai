import itertools

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
import gensim.downloader
import optuna
import pickle
import os
import time
from torch.utils import data
from tqdm import tqdm
import torch.optim
from chu_liu_edmonds import decode_mst

WORD_EMBED_SIZE = 100
POS_EMBED_SIZE = 50
NUM_EPOCHS = 75

POS_VOCAB_SIZE = 45  # calculated using possible_pos_tags = count_possible_pos_tags(train_sent + test_sent + comp_sent)
HIDDEN_DIM = 256
MAX_SENTENCE_LENGHT = 100
CHU_LIU_EVERY = 1
MINI_FLAG = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
torch.manual_seed(1)


class DnnPosTagger(nn.Module):
    def __init__(self, pos_2_index):
        super(DnnPosTagger, self).__init__()

        self.fc1 = nn.Linear(HIDDEN_DIM * MAX_SENTENCE_LENGHT * 2, HIDDEN_DIM * MAX_SENTENCE_LENGHT // 2)
        self.fc2 = nn.Linear(HIDDEN_DIM * MAX_SENTENCE_LENGHT // 2, (MAX_SENTENCE_LENGHT + 1) * (MAX_SENTENCE_LENGHT))
        # self.softmax = nn.Softmax(dim=1)

        self.POS_embedding = nn.Embedding(POS_VOCAB_SIZE + 1, POS_EMBED_SIZE)
        self.lstm = nn.LSTM(input_size=WORD_EMBED_SIZE + POS_EMBED_SIZE, hidden_size=HIDDEN_DIM, num_layers=4,
                            bidirectional=True,
                            batch_first=True)
        self.Relu = nn.ReLU()
        self.hidden2score = nn.Sequential(self.fc1, self.Relu, self.fc2)
        self.pos_2_index = pos_2_index

    def forward(self, word_embeds_tensor, pos_idx_tensor):
        pos_embeds = self.POS_embedding(pos_idx_tensor)  # [batch_size, seq_length, emb_dim]
        embeds = torch.cat((word_embeds_tensor, pos_embeds), dim=1)
        lstm_out, _ = self.lstm(embeds)  # [batch_size, seq_length, 2*hidden_dim]
        score_matrices = self.hidden2score(lstm_out.flatten(1)).view(-1, MAX_SENTENCE_LENGHT, MAX_SENTENCE_LENGHT + 1)
        # [MAX_SENTENCE_LENGHT, MAX_SENTENCE_LENGHT]

        return score_matrices


class PosTagFC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PosTagFC, self).__init__()
        self.layer1 = nn.Linear(input_size, 2 * hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(self.relu(x))
        x = self.relu(self.dropout2(self.layer2(x)))
        x = self.layer3(x)
        return x


class NewDnnPosTagger(nn.Module):
    def __init__(self, pos_2_index):
        super(NewDnnPosTagger, self).__init__()

        self.fc = PosTagFC(input_size=4 * HIDDEN_DIM, hidden_size=HIDDEN_DIM, output_size=1)

        # self.fc1 = nn.Linear(HIDDEN_DIM * MAX_SENTENCE_LENGHT * 2, HIDDEN_DIM * MAX_SENTENCE_LENGHT // 2)
        # self.fc2 = nn.Linear(HIDDEN_DIM * MAX_SENTENCE_LENGHT // 2, (MAX_SENTENCE_LENGHT + 1) * (MAX_SENTENCE_LENGHT))
        # self.softmax = nn.Softmax(dim=1)

        self.POS_embedding = nn.Embedding(POS_VOCAB_SIZE + 1, POS_EMBED_SIZE)
        self.lstm = nn.LSTM(input_size=WORD_EMBED_SIZE + POS_EMBED_SIZE, hidden_size=HIDDEN_DIM, num_layers=4,
                            bidirectional=True,
                            batch_first=True, dropout=0.4)
        self.Relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.3)
        # self.hidden2score = nn.Sequential(self.fc1, self.Relu, self.fc2)
        self.pos_2_index = pos_2_index

    def forward(self, word_embeds_tensor, pos_idx_tensor, sen_lens):
        pos_embeds = self.POS_embedding(pos_idx_tensor)  # [batch_size, seq_length, emb_dim]
        embeds = torch.cat((word_embeds_tensor, pos_embeds), dim=2)
        lstm_out, _ = self.lstm(embeds)  # [batch_size, seq_length, 2*hidden_dim]

        score_matrices = torch.zeros(word_embeds_tensor.shape[0], MAX_SENTENCE_LENGHT + 1, MAX_SENTENCE_LENGHT + 1).to(
            device)
        for i in range(word_embeds_tensor.shape[0]):
            first = torch.vstack([row.repeat(sen_lens[i] + 1, 1) for row in lstm_out[i][:sen_lens[i], :]])
            rooted_sentence = torch.vstack((torch.zeros(1, 2 * HIDDEN_DIM).to(device), lstm_out[i][:sen_lens[i], :]))
            second = rooted_sentence.repeat(sen_lens[i], 1)
            fc_input = torch.hstack((first, second))
            # fc_input = torch.vstack(
            #     [torch.cat(pair, 0) for pair in itertools.product(rooted_sentence, rooted_sentence)])
            fc_output = self.fc(fc_input).fill_diagonal_(0).reshape((sen_lens[i], sen_lens[i] + 1))
            score_matrices[i][:sen_lens[i], :sen_lens[i] + 1] = fc_output[:sen_lens[i], :sen_lens[i] + 1]
        # [MAX_SENTENCE_LENGHT, MAX_SENTENCE_LENGHT]

        return score_matrices


# class DependencyParser(nn.Module):
#     def __init__(self, *args):
#         super(DependencyParser, self).__init__()
#         self.word_embedding =  gensim.downloader.load(f'glove-twitter-{EMBED_SIZE}')
#         self.hidden_dim = self.word_embedding.embedding_dim
#         self.encoder = # Implement BiLSTM module which is fed with word embeddings and o
#         self.edge_scorer = # Implement a sub-module to calculate the scores for all poss
#         self.loss_function = # Implement the loss function described above
#     def forward(self, sentence):
#         word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence
#         # Pass word_idx through their embedding layer
#         # Get Bi-LSTM hidden representation for each word in sentence
#         # Get score for each possible edge in the parsing graph, construct score matrix
#         # Calculate the negative log likelihood loss described above
#         return loss, score_mat

def build_truth_score_tensor(sentence):
    """takes a sentence and builds a matrix of the true scores"""
    score_tensor = torch.zeros((MAX_SENTENCE_LENGHT, MAX_SENTENCE_LENGHT))
    if len(sentence) > MAX_SENTENCE_LENGHT:
        return score_tensor

    for i, word in enumerate(sentence):
        score_tensor[i][int(word[4]) - 1] = 1
    return score_tensor


def build_dir_structure():
    """makes sure the directory structure is in place"""
    if not os.path.exists("generated_files"):
        os.mkdir("generated_files")
    if not os.path.exists("generated_files/studies"):
        os.mkdir("generated_files/studies")


def create_sentence_list(path, eval = False, mark_OOV = False):
    """creates a list of sentences from a file"""
    tic = time.time()
    build_dir_structure()
    if os.path.exists(f"generated_files\{path}.embedded"):
        # load the pickle file
        sentences = pickle.load(open(f"generated_files\{path}.embedded", "rb"))
        print(f"loaded {path} from pickle file, time taken {time.time() - tic}")
        return sentences

    with open(path, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        sentences = []
        sentence = []
        word_embedding = gensim.downloader.load(f'glove-wiki-gigaword-{WORD_EMBED_SIZE}')
        for i, line in enumerate(lines):
            if line == '\n':
                sentences.append(sentence)
                sentence = []
            else:
                line = line.split("\t")
                del line[7]
                if line[1] not in word_embedding and mark_OOV:
                    line[1] = line[1] + 'unk'

                line = [int(line[0]), line[1], word_embedding[line[1].lower() if line[1] in word_embedding else 'unk'], line[3], 0 if eval else min(int(line[6]), MAX_SENTENCE_LENGHT)]
                sentence.append(line)
    f.close()
    print(f"generated {path}.embedded sentences from file, time taken {time.time() - tic}")
    pickle.dump(sentences, open(f"generated_files\{path}.embedded", "wb"))
    print(f"saved {path}.embedded to a pickle file, time taken {time.time() - tic}")
    return sentences


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_possible_pos_tags(sentences):
    """count the number of possible pos tags"""
    pos_tags = set()
    for sentence in sentences:
        for word in sentence:
            pos_tags.add(word[3])
    return len(pos_tags)


def pad_sentence(sentence):
    """pads the sentence with zeros"""
    if len(sentence) < MAX_SENTENCE_LENGHT:
        for i in range(MAX_SENTENCE_LENGHT - len(sentence)):
            sentence.append([0, 0, [0] * MAX_SENTENCE_LENGHT, 0, 0])
    for i in range(MAX_SENTENCE_LENGHT):
        sentence[i][2] = torch.tensor(sentence[i][2])
    return sentence


def sentence_to_tensor(sentence, pos_2_idx):
    """converts a sentence to a tensor"""
    word_emb_tensor = sentence[0][2].clone().detach().reshape(1, WORD_EMBED_SIZE)
    pos_idx_tensor = [pos_2_idx[sentence[0][3]]]
    for i in range(1, len(sentence)):
        word_emb_tensor = torch.cat((word_emb_tensor, torch.tensor(sentence[i][2]).reshape(1, WORD_EMBED_SIZE)), dim=0)
        pos_idx_tensor.append(pos_2_idx[sentence[i][3]])
    pos_idx_tensor = torch.tensor(pos_idx_tensor)
    return word_emb_tensor, pos_idx_tensor


def build_data_structs(path, pos_2_idx, mini=False, eval = False, mark_OOV = False):
    """builds the data structures needed for training and evaluating"""
    tic = time.time()
    if os.path.exists(f"generated_files\{path}{'.markOOV' if mark_OOV else ''}{'.eval' if eval else ''}.fully_tensored"):
        # load the pickle file
        loaded_pos_2_idx, sentences = pickle.load(open(f"generated_files\{path}{'.markOOV' if mark_OOV else ''}{'.eval' if eval else ''}.fully_tensored", "rb"))
        if pos_2_idx == pos_2_idx:
            print(f"loaded {path}{'.markOOV' if mark_OOV else ''}{'.eval' if eval else ''}.fully_tensored from pickle file, time taken {time.time() - tic}")
            return sentences if not mini else sentences[::10]

    sentences = create_sentence_list(path, eval=eval, mark_OOV = mark_OOV)
    sentences = [sentence for sentence in sentences if len(sentence) <= MAX_SENTENCE_LENGHT]
    sen_lens = [torch.tensor(min(len(sentence), MAX_SENTENCE_LENGHT)) for sentence in sentences]
    padded_sentences = [pad_sentence(sentence) for sentence in sentences]
    x = [sentence_to_tensor(sentence, pos_2_idx) for sentence in padded_sentences]
    # y = [build_truth_score_tensor(sentence) for sentence in sentences]
    y = [torch.tensor([int(word[4]) for word in sentence[:MAX_SENTENCE_LENGHT]]) for sentence in sentences]
    tensored_sentences = list(zip(zip(x, y, sen_lens), sentences)) if eval else list(zip(x, y, sen_lens))
    print("generated tensored sentences, time taken", time.time() - tic)
    pickle.dump((pos_2_idx, tensored_sentences), open(f"generated_files\{path}{'.markOOV' if mark_OOV else ''}{'.eval' if eval else ''}.fully_tensored", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved tensored sentences to a pickle file {path}{'.markOOV' if mark_OOV else ''}{'.eval' if eval else ''}.fully_tensored , time taken", time.time() - tic)
    return tensored_sentences if not mini else tensored_sentences[::10]


class CustomDataset(data.Dataset):
    def __init__(self, path, pos_2_idx, mini=False):
        self.data = build_data_structs(path, pos_2_idx, mini)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def generate_pos_2_idx():
    """generates a dictionary of pos tags to idx"""
    sentences = create_sentence_list("train.labeled")
    pos_2_idx = {}
    for sentence in sentences:
        for word in sentence:
            if word[3] not in pos_2_idx:
                pos_2_idx[word[3]] = len(pos_2_idx)
    pos_2_idx[0] = len(pos_2_idx)
    return pos_2_idx


def show_compare_graph(title, train_values, valid_values):
    """
    plot the losses
    """
    plt.figure(f"{title} as a function of epochs", figsize=(8, 8))
    plt.plot(range(1, len(train_values) + 1), train_values, label="train")
    plt.plot(range(1, len(valid_values) + 1), valid_values, label="validation")
    plt.title(title)
    plt.legend()
    plt.savefig(f"generated_files\{title}_graph.png")


def decode_sentences(preds, sent_lens):
    """uses the chu liu edmonds algorithm to decode the sentences
    using the imported method decode_mst"""
    decoded_sentences = []
    for pred, sen_len in zip(preds, sent_lens):
        dec = decode_mst(insert_zeros(pred, [0]).detach().cpu().T, sen_len + 1, has_labels=False)[0]
        decoded_sentences.append(
            torch.from_numpy(dec).to(device)[1: sen_len + 1])
        # decoded_sentences.append(
        #     torch.argmax(input=pred[:sen_len.item(), :sen_len.item()+1], dim=1).clone().detach().cpu())
    return decoded_sentences


def batch_sentences(sentences, batchsize, shuffle=True, eval=False):
    """batch sentences, drop last batch
    shuffle = True"""
    if shuffle:
        r = torch.randperm(len(sentences))
        sentences = [sentences[i] for i in r]
    WORD_embs, POS_idxs, Ys, SENS_lens, SENTenses = [], [], [], [], []
    if eval:
        sentences, uncoded_sentences = zip(*sentences)
    all_x, all_y, all_sen_lens = zip(*sentences)
    all_word_emb, all_pos_idx = zip(*all_x)
    for i in range(0, len(sentences), batchsize):
        word_embs = torch.cat([word_emb.unsqueeze(0).to(device=device) for word_emb in all_word_emb[i:i + batchsize]],
                              dim=0)
        WORD_embs.append(word_embs)
        pos_idxs = torch.cat([pos_idxs.unsqueeze(0).to(device=device) for pos_idxs in all_pos_idx[i:i + batchsize]],
                             dim=0)
        POS_idxs.append(pos_idxs)
        Y = torch.cat([y.unsqueeze(0).to(device=device) for y in all_y[i:i + batchsize]], dim=0)
        Ys.append(Y)
        SEN_lens = all_sen_lens[i:i + batchsize]
        SENS_lens.append(torch.tensor(SEN_lens, device=device))
        if eval:
            SENTenses.append(uncoded_sentences[i:i + batchsize])
    Xs = list(zip(WORD_embs, POS_idxs))
    if eval:
        return list(zip(zip(Xs, Ys, SENS_lens), SENTenses))
    return list(zip(Xs, Ys, SENS_lens))[:-1:]


def calculate_CE_loss(preds, ys, sen_lens):
    """calculate the cross entropy loss"""
    loss = 0
    for pred, y, sen_len in zip(preds, ys, sen_lens):
        # for i in range(sen_len):
        prediction = pred[:sen_len, :sen_len + 1].to(device)
        target = y[:sen_len].to(device)
        loss += F.cross_entropy(prediction, target) / sen_len

    return loss


"""
######         needs to be fixed before use           ######

def sum_loss(batch_size, sen_lens, cut_outputs, cut_labels):

    loss =0
    for j in range(batch_size):
        for k in range(sen_lens[j]):
            cut_labels[j][k][k] = cut_outputs[j][k][k]

    for j in range(batch_size):
        loss += criterion(cut_outputs[j], cut_labels[j])
    return loss
"""


def insert_zeros(x, all_j):
    """
    insert zeros to the tensor x in the positions all_j
    taken from https://stackoverflow.com/questions/74423476/appending-zero-rows-to-a-2d-tensor-in-pytorch
    :param x:
    :param all_j:
    :return:
    """
    zeros_ = torch.zeros_like(x[:1])
    pieces = []
    i = 0
    for j in all_j + [len(x)]:
        pieces.extend([x[i:j], zeros_])
        i = j
    return torch.cat(pieces[:-1], dim=0)


def eval_first_sentence(sentence, sen_len, target):
    """evaluates the first sentence of the dev set"""
    cut_target = target[:sen_len]
    preds = torch.argmax(sentence[:sen_len, :sen_len+1], dim=1)
    preds_chu = decode_sentences([sentence], [sen_len])[0]
    # preds_chu = decode_mst(sentence.detach().cpu(), sen_len+1, has_labels = False)[0][1:sen_len+1]
    print("\n")
    print("------------ FIRST SENTENCE EVAL ------------")
    print("Preds: ", preds)
    print("chuli: ", preds_chu)
    print("Target: ", cut_target)
    print(f"argmax Acc: {torch.sum(preds == cut_target) / sen_len}")
    print(f"bing chillin Acc: {torch.sum(torch.tensor(preds_chu, device=device) == cut_target) / sen_len}")


def search_hyperparams():
    """uses optuna to search the best hyper parameters, uses the run_experiment function"""
    study_name = "test_study"
    storage_name = "sqlite:///generated_files/studies{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    study.trials_dataframe().to_csv(f"generated_files/studies/{study_name}.csv")
    study.optimize(run_experiment, n_trials=100)


def run_experiment(trial):
    """a wrapper function for the objective function using the trial"""
    optimizers_dict = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
    loss_dict = {"CE": calculate_CE_loss}
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    batchsize = trial.suggest_int("batch_size", 8, 256, log=True)
    weight_dec = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer = optimizers_dict[trial.suggest_categorical("optimizer", tuple(optimizers_dict.keys()))]
    loss_function = loss_dict[trial.suggest_categorical("loss_function", tuple(loss_dict.keys()))]
    best_train_acc, best_test_acc = objective(batchsize, learning_rate, weight_dec, optimizer, loss_function, trial)
    return 1 - best_test_acc


def objective(batch_size, learning_rate, weight_dec, optimizer_model, loss_function, trial=None):
    """objective function for optuna"""
    pos_2_idx = generate_pos_2_idx()
    model = NewDnnPosTagger(pos_2_idx).to(device)
    # kwargs = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True, 'drop_last': True}
    total_predictions = 0

    optimizer = optimizer_model(model.parameters(), lr=learning_rate, weight_decay=weight_dec)

    train_sentences = build_data_structs("train.labeled", pos_2_idx, mini=MINI_FLAG)
    test_sentences = build_data_structs("test.labeled", pos_2_idx, mini=False)
    best_test_acc = 0
    best_train_acc = 0
    max_acc = 0

    # print number of parameters in the model
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    total_losses = {"Train": [], "Test": []}
    for epoch in range(NUM_EPOCHS):
        ##################
        # train the model#
        ##################
        total_loss = {"Train": 0, "Test": 0}
        true_classified = 0
        for i, ((word_embeds, pos_embeds), labels, sen_lens) in enumerate(
                tq := tqdm(batch_sentences(train_sentences, batch_size), leave=True)):
            model.train()
            # Forward pass
            outputs = model(word_embeds, pos_embeds, sen_lens)
            loss = loss_function(outputs, labels, sen_lens)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss["Train"] += loss.item() / batch_size
            if i == 0:
                eval_first_sentence(outputs[0], sen_lens[0], labels[0])
            if i % CHU_LIU_EVERY != 0:
                decoded = decode_sentences(outputs, sen_lens)
                true_classified += sum(
                    [torch.sum(decoded_sent.clone().to(device) == decoded_label[:sen_len]) for
                     decoded_sent, decoded_label, sen_len in zip(decoded, labels, sen_lens)])
                total_predictions += sum(sen_lens)
            del outputs, loss, word_embeds, pos_embeds, labels, sen_lens
            torch.cuda.empty_cache()
            tq.set_description(f'Epoch {epoch + 1}/{NUM_EPOCHS}\tTrain Loss: {total_loss["Train"] / (i + 1):.3f}\t')
        # train_accuracy = true_classified / total_predictions
        train_accuracy = 0
        ##################
        # eval the model #
        ##################
        total_predictions = 0
        true_classified = 0
        for i, ((word_embeds, pos_embeds), labels, sen_lens) in enumerate(
                tq := tqdm(batch_sentences(test_sentences, batch_size), leave=True)):
            model.eval()
            loss = 0
            with torch.no_grad():
                # Forward pass
                outputs = model(word_embeds, pos_embeds, sen_lens)

                loss += loss_function(outputs, labels, sen_lens)
                if i % CHU_LIU_EVERY == 0:
                    decoded = decode_sentences(outputs, sen_lens)
                    true_classified += sum([torch.sum(
                        decoded_sent.clone().to(device) == decoded_label[:sen_len]) for
                        decoded_sent, decoded_label, sen_len in zip(decoded, labels, sen_lens)])
                    total_predictions += sum(sen_lens)
            total_loss["Test"] += loss.item() / batch_size

            del outputs, loss, word_embeds, pos_embeds, labels, sen_lens
            torch.cuda.empty_cache()
            tq.set_description(f'Epoch {epoch + 1}/{NUM_EPOCHS}\tTest Loss: {total_loss["Test"] / (i + 1):.3f}\t')
        test_accuracy = true_classified / total_predictions

        best_test_acc = max(best_test_acc, test_accuracy)
        best_train_acc = max(best_train_acc, train_accuracy)

        # save best model
        if best_test_acc == test_accuracy:
            torch.save(model.state_dict(), "generated_files/best_model.pt")

        if trial != None:
            trial.report(test_accuracy, epoch)

        print(f"train acc:{train_accuracy:.3f}\ttest acc:{test_accuracy:.3f}, best test acc:{best_test_acc:.3f}")
        total_losses["Train"].append(total_loss["Train"])
        total_losses["Test"].append(total_loss["Test"])
    # clean the memory to avoid overflow after a long sequence of trials
    del model, optimizer, train_sentences, test_sentences, total_loss
    torch.cuda.empty_cache()

    return best_train_acc, best_test_acc

def check_train_acc():
    pos_2_idx = generate_pos_2_idx()
    model = NewDnnPosTagger(pos_2_idx).to(device)
    model.load_state_dict(torch.load("generated_files/best_model.pt"))

    test_sentences = build_data_structs("train.labeled", pos_2_idx, mini=False)
    total_predictions = 0
    true_classified = 0
    for i, ((word_embeds, pos_embeds), labels, sen_lens) in enumerate(
            tq := tqdm(batch_sentences(test_sentences, batch_size), leave=True)):
        model.eval()
        with torch.no_grad():
            # Forward pass
            outputs = model(word_embeds, pos_embeds, sen_lens)

            decoded = decode_sentences(outputs, sen_lens)
            true_classified += sum([torch.sum(
                decoded_sent.clone().to(device) == decoded_label[:sen_len]) for
                decoded_sent, decoded_label, sen_len in zip(decoded, labels, sen_lens)])
            total_predictions += sum(sen_lens)

        del outputs, word_embeds, pos_embeds, labels, sen_lens
        torch.cuda.empty_cache()

    test_accuracy = true_classified / total_predictions

    print(f"test acc:{test_accuracy:.3f}")

def load_best_model():
    pos_2_idx = generate_pos_2_idx()
    model = NewDnnPosTagger(pos_2_idx).to(device)
    model.load_state_dict(torch.load("generated_files/best_model.pt"))
    return model
def load_and_save_predictions(path, mark_OOV):
    """loads a file from path, and saves the predictions to a file with the same name but with .pred extension
    in the file format"""
    model = load_best_model()
    idx_2_pos = {v: k for k, v in model.pos_2_index.items()}
    test_sentences = build_data_structs(path, model.pos_2_index, mini=False, eval=True, mark_OOV=mark_OOV)
    with open(path + ".pred", "w") as f:
        for i, (((word_embeds, sentence_pos_indexes), labels, sen_lens), cur_batch_sentences) in enumerate(
                tq := tqdm(batch_sentences(test_sentences, 100, shuffle=False, eval=True), leave=True)):
            with torch.no_grad():
                # Forward pass
                outputs = model(word_embeds, sentence_pos_indexes, sen_lens)

                decoded = decode_sentences(outputs, sen_lens)
                for decoded_sent, sen_len, sentence, cur_pos_indexes in zip(decoded, sen_lens, cur_batch_sentences, sentence_pos_indexes):
                    for word_idx, (decoded_word, word, pos_idx) in enumerate(zip(decoded_sent, sentence, cur_pos_indexes)):
                        f.write(f"{word_idx+1}\t{word[1]}\t_\t{idx_2_pos[pos_idx.item()]}\t_\t_\t{decoded_word}\t_\t_\t_\n")
                    f.write("\n")

def develop():
    batch_size = 64
    lr = 0.0025
    weight_decay = 0.0075
    optim_model = torch.optim.AdamW
    loss = calculate_CE_loss
    objective(batch_size=batch_size, learning_rate=lr, weight_dec=weight_decay,
              optimizer_model=optim_model, loss_function=loss)
    # check_train_acc()
    # search_hyperparams()
    # Hyperparameters

if __name__ == "__main__":
    # set torch into benchmark mode
    build_dir_structure()
    load_and_save_predictions("test.labeled", mark_OOV=True)
