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


def clean_corpus(tokenized_corpus, stop_word=stopwords.words('english')):
    cleaned_corpus = []
    for tokens in tokenized_corpus:
        tmp = []
        for token in tokens:
            cleaned_token = remove_characters(token)
            token = cleaned_token.strip()
            if token is not None and token not in stop_word:
                tmp.append(cleaned_token)
        cleaned_corpus.append(tmp)
    return cleaned_corpus


def casefolding(tokenized_text, to="lower"):
    if to == "lower":
        return [[t.lower() for t in data] for data in tokenized_text]
    elif to == "upper":
        return [[t.upper() for t in data] for data in tokenized_text]


def split_data(
    dataset, train_split=72, test_split=20, valid_split=8, seed=148301
        ):

    assert train_split + test_split + valid_split == 100
    X, y = dataset
    init_train_split = train_split+valid_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=init_train_split/100, test_size=test_split/100,
        random_state=seed
    )
    X_val = []
    y_val = []
    train_split /= init_train_split
    valid_split /= init_train_split

    if valid_split > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, train_size=train_split,
            test_size=valid_split, random_state=seed
        )

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)
