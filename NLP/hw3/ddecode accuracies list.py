import pickle
from main import show_compare_graph

def main():
    # load the accuracies list
    accuracies = pickle.load(open('generated_files/accuracy_and_loss_lists', 'rb'))

    # print the accuracies list
    total_losses = accuracies['total_losses']
    UAS = accuracies['accuracies']
    max_epoch = 100
    show_compare_graph("Losses", total_losses["Train"][:max_epoch], total_losses["Test"][:max_epoch])
    show_compare_graph("UAS", [a.item() for a in UAS["Train"]][:max_epoch], [a.item() for a in UAS["Test"]][:max_epoch])
    print(accuracies)

main()