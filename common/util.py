import string
# import re
from nltk.corpus import stopwords


def remove_characters(text, charset=string.punctuation):
    """
    Remove a set of character from text
    :param text: string, text input
    :param charset: string, sequence of characters that will be removed
    """
    return text.translate(str.maketrans('', '', charset))


def remove_words(tokenized_text, wordset=stopwords.words('english')):
    return list(filter(lambda x: x not in wordset, tokenized_text))
