import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test, memm_viterbi


def main():
    threshold = 1
    lam = 1
    model = 1

    train_path = f"data/train{model}.wtag"
    # test_path = "data/comp1.words"
    test_path = f"data/comp{model}.words"
    weights_path = f'weights_{model}.pkl'
    predictions_path = f'comp_m1_123456789_987654321.wtag'

    # statistics, feature2id = preprocess_train(train_path, threshold)
    # get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(pre_trained_weights)
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)


if __name__ == '__main__':
    main()
