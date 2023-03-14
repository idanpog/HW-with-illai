"""
this file is responsible for the data management of the application
loads the data, handles the preprocessing and the data augmentation
"""

def load_dataset(path):
    """
    loads the data from the given path
    returns a list of tuples of the form (paragraph, translated)
    :param self:
    :param path:
    :return:
    """
    with open(path, 'r+', encoding='iso-8859-15') as f:
        lines = f.readlines()
    lines = "".join(lines)
    lines = lines.split("German:\n")
    a = [line.split("English:\n") for line in lines if line != ""]
    return [tuple(x) for x in a]


def check_max_length(path):
    data = load_dataset(path)
    print(max([len(x[0]) for x in data]))


def preprocess_function(data, tokenizer):
    prefix = 'Translate German to English: '

    inputs = [prefix + par[0] for par in data]
    targets = [par[1] for par in data]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=1500, truncation=True)
    return model_inputs