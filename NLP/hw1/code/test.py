import pickle
import pandas as pd
from preprocessing import read_test
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt


def calc_accuracy(true_path, predictions_path, tags):
    true = read_test(true_path, tags)
    pred = read_test(predictions_path, tags)
    all_tags, correct = 0, 0
    for t_sen, p_sen in zip(true, pred):
        t_tags = t_sen[1][2:-1]
        p_tags = p_sen[1][2:-1]
        for t_tag, p_tag in zip(t_tags, p_tags):
            all_tags += 1
            correct += (1 if t_tag == p_tag else 0)
    return correct / all_tags


def plot_confusion_matrix(true_path, pred_path, tags):
    true = read_test(true_path, tags)
    pred = read_test(pred_path, tags)
    false = defaultdict(int)
    worst_ten = {}
    confusion = {tag: defaultdict(int) for tag in tags}
    for t_sen, p_sen in zip(true, pred):
        t_tags = t_sen[1][2:-1]
        p_tags = p_sen[1][2:-1]
        for t_tag, p_tag in zip(t_tags, p_tags):
            if p_tag in confusion[t_tag]:
                confusion[t_tag][p_tag] += 1
            else:
                confusion[t_tag][p_tag] = 1

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


if __name__ == '__main__':
    with open('weights1.pkl', 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    possible_tags = list(feature2id.feature_statistics.tags)
    # acc_conf("data/test1.wtag", 'predictions.wtag', possible_tags)
    # plot_confusion_matrix("data/test1.wtag", 'predictions.wtag', possible_tags)
    print(calc_accuracy("data/test1.wtag", 'predictions.wtag', possible_tags))
