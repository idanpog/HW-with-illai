from collections import defaultdict
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

WORD = 0
TAG = 1


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]]) \
        -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features that are relevant to the given history
    """
    c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word = history
    features = []

    # f100
    if (c_word, c_tag) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_word, c_tag)])
    for i in range(1, min(4, len(c_word) - 1)):
        # f101
        if (c_word[-i:], c_tag) in dict_of_dicts["f101"]:
            features.append(dict_of_dicts["f101"][(c_word[-i:], c_tag)])
        # f102
        if (c_word[:i], c_tag) in dict_of_dicts["f102"]:
            features.append(dict_of_dicts["f102"][(c_word[:i], c_tag)])
    # f103
    if (pp_tag, p_tag, c_tag) in dict_of_dicts["f103"]:
        features.append(dict_of_dicts["f103"][(pp_tag, p_tag, c_tag)])
    # f104
    if (p_tag, c_tag) in dict_of_dicts["f104"]:
        features.append(dict_of_dicts["f104"][(p_tag, c_tag)])
    # f105
    if c_tag in dict_of_dicts["f105"]:
        features.append(dict_of_dicts["f105"][c_tag])
    # f106
    if (p_word, c_tag) in dict_of_dicts["f106"]:
        features.append(dict_of_dicts["f106"][(p_word, c_tag)])
    # f107
    if (c_tag, n_word) in dict_of_dicts["f107"]:
        features.append(dict_of_dicts["f107"][(c_tag, n_word)])
    for feat in ["fcap", "fnum"]:
        if c_tag in dict_of_dicts[feat]:
            features.append(dict_of_dicts[feat][c_tag])
    return features


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences


def memm_viterbi(sentence, pre_trained_weights, feature2id, tops, bottoms):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    ts = []
    n = len(sentence)
    beam_width = 2
    pi = {}
    bp = {}
    tags = feature2id.feature_statistics.tags.union("*")
    beam = list(tags)
    p_beam = list(tags)
    pi[1] = {}
    pi[1][("*", "*")] = 1
    for k in range(2, n - 1):
        pi[k] = {}
        bp[k] = {}
        for u in beam:
            for v in tags:
                M = 0
                curr_bp = None
                for t in p_beam:
                    if (t, u) in pi[k - 1]:
                        history = (sentence[k], v, sentence[k - 1], u, sentence[k - 2], t, sentence[k + 1])
                        hist = (sentence[k], sentence[k - 1], u, sentence[k - 2], t, sentence[k + 1])
                        if hist in bottoms:
                            curr = pi[k - 1][(t, u)] * tops[history] / bottoms[hist]
                        else:
                            bottoms[hist] = 0
                            tops[history] = np.exp(sum([pre_trained_weights[j] for j in
                                                        represent_input_with_features(history,
                                                                                      feature2id.feature_to_idx)]))
                            for tag in tags:
                                alt = (sentence[k], tag, sentence[k - 1], u, sentence[k - 2], t, sentence[k + 1])
                                if alt not in tops:
                                    tops[alt] = np.exp(sum([pre_trained_weights[j] for j in
                                                            represent_input_with_features(alt,
                                                                                          feature2id.feature_to_idx)]))
                                bottoms[hist] += tops[alt]
                            curr = pi[k - 1][(t, u)] * tops[history] / bottoms[hist]
                        if curr > M:
                            M = curr
                            curr_bp = t
                    pi[k][(u, v)] = M
                    bp[k][(u, v)] = curr_bp
        beam = sorted(pi[k].items(), key=lambda item: item[1])[-beam_width:]
        p_beam = [tup[0][0] for tup in beam]
        beam = [tup[0][1] for tup in beam]

    t_n1, t_n = sorted(pi[n - 2].items(), key=lambda item: item[1], reverse=True)[0][0]
    ts.insert(0, t_n)
    ts.insert(0, t_n1)
    for k in range(n - 4, 1, -1):
        ts.insert(0, bp[k + 2][(ts[0], ts[1])])
    return ts


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")
    tops, bottoms = {}, {}

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id, tops, bottoms)
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()


def calc_accuracy(true_path, predictions_path, tags):
    true = read_test(true_path)
    pred = read_test(predictions_path)
    all_tags, correct = 0, 0
    for t_sen, p_sen in zip(true, pred):
        t_tags = t_sen[1][2:-1]
        p_tags = p_sen[1][2:-1]
        for t_tag, p_tag in zip(t_tags, p_tags):
            all_tags += 1
            true += int(t_tag == p_tag)
    return correct / all_tags


def plot_confusion_matrix(true_path, pred_path, tags):
    true = read_test(true_path)
    pred = read_test(pred_path)
    confusion = {tag: defaultdict(int) for tag in tags}
    false = defaultdict(int)
    worst_ten = {}
    for tag in tags:
        false[tag] = sum([confusion[tag][o_tag] for o_tag in tags if o_tag != tag])
        v = list(false.values())
        k = list(false.keys())
    for i in range(10):
        worst = k[v.index(max(v))]
        k.pop(v.index(max(v)))
        v.pop(v.index(max(v)))
        worst_ten[worst] = confusion[worst]
    data = pd.DataFrame(worst_ten)
    sns.heatmap(data)
    plt.show()


def main():
    test_path = "data/test1.wtag"
    weights_path = 'weights.pkl'
    predictions_path = 'predictions.wtag'
    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)


if __name__ == '__main__':
    main()
