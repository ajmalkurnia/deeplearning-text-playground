import string
# import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


def remove_characters(text, charset=string.punctuation):
    """
    Remove a set of character from text
    :param text: string, text input
    :param charset: string, sequence of characters that will be removed
    """
    return text.translate(str.maketrans('', '', charset))


def remove_words(tokenized_text, wordset=stopwords.words('english')):
    return list(filter(lambda x: x not in wordset, tokenized_text))


def clean_text(tokenized_corpus, stopwrods=stopwords.words('english')):
    cleaned_corpus = []
    for tokens in tokenized_corpus:
        tmp = []
        for token in tokens:
            cleaned_token = remove_characters(token)
            if cleaned_token.strip():
                tmp.append(cleaned_token)
        cleaned_corpus.append(remove_words(tmp, stopwords))
    return cleaned_corpus


def casefolding(tokenized_text, to="lower"):
    if to == "lower":
        return [[t.lower() for t in data] for data in tokenized_text]
    elif to == "upper":
        return [[t.upper() for t in data] for data in tokenized_text]


def split_dataset(
    dataset, train_split=0.72, test_split=0.2, valid_split=0.08, seed=148301
        ):

    assert train_split + test_split + valid_split == 1
    X, y = dataset
    init_train_split = train_split+valid_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=init_train_split, test_size=test_split,
        random_state=seed
    )

    train_split /= init_train_split
    valid_split /= init_train_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, train_size=train_split, test_size=valid_split,
        random_state=seed
    )

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)
