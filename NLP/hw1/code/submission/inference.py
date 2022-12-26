from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import scipy.sparse
from time import time


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
                    if (t, u) in pi[k-1]:
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
                        if curr_bp == "PRP$":
                            x=3
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


def q_prob(c_tag, p_tag, pp_tag, sentence, i, pre_trained_weights, feature2id):
    history = (sentence[i], c_tag, sentence[i - 1], p_tag, sentence[i - 2], pp_tag, sentence[i + 1])
    top = sum([pre_trained_weights[j] for j in
               represent_input_with_features(history, feature2id.feature_to_idx)])
    # switch_tag = lambda x: [sentence[i], x, sentence[i - 1], p_tag, sentence[i - 2], pp_tag, sentence[i + 1]]
    # bottom = sum([np.exp(sum([pre_trained_weights[i] for i in
    #                           feature2id.represent_input_with_features(switch_tag(c_tag_tag),
    #                                                                    feature2id.feature_to_idx)]))
    #               for c_tag_tag in tags])
    # return top / bottom
    return top


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, feature2id.feature_statistics.tags, tagged=tagged, skip_tags=True)

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
