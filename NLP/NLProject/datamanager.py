"""
this file is responsible for the data management of the application
loads the data, handles the preprocessing and the data augmentation
"""




def load_train():
    """
    loads the data from the given path
    returns a list of tuples of the form (paragraph, translated)
    :param self:
    :param path:
    :return:
    """
    with open('train.labeled', 'r+', encoding='iso-8859-15') as f:
        lines = f.readlines()
    lines = "".join(lines)
    lines = lines.split("German:\n")
    a = [line.split("English:\n") for line in lines if line != ""]
    return [tuple(x) for x in a]

a = load_train()
