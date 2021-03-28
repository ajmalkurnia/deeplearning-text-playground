from nltk.corpus import stopwords
import pandas as pd

import json
from glob import glob
from common import utils
from common import tokenization
DATA_DIR = "../resources/dataset/"


class Dataset():
    @staticmethod
    def open_indosum(path):
        data = []
        with open(path, "r") as jsonf:
            for line in jsonf:
                raw_data = json.loads(line)
                data.append({
                    "text": raw_data["paragraphs"],
                    "label": raw_data["category"]
                })
        return pd.DataFrame(data)

    @staticmethod
    def open_postag_id(path):
        data = []
        with open(path, "r") as f:
            sentences = f.read().split("\n</kalimat>")
            for sentence in sentences:
                token_sequence = []
                tag_sequenece = []
                tokens = sentence.split("\n")
                for idx, token in enumerate(tokens):
                    if len(token.split("\t")) == 2:
                        word, tag = token.split("\t")
                        token_sequence.append(word)
                        tag_sequenece.append(tag)
                data.append({
                    "tokenized_text": token_sequence,
                    "tag": tag_sequenece
                })
        return data

    @staticmethod
    def open_postag_ud(path):
        data = []
        token_sequence = []
        tag_sequence = []
        with open(path, "r") as f:
            for line in f:
                if line[0] == "#":
                    continue
                elif line.strip() != "":
                    words_detail = line.strip().split("\t")
                    token_sequence.append(words_detail[1])
                    tag_sequence.append(words_detail[3])
                else:
                    data.append({
                        "tokenized_text": token_sequence,
                        "tag": tag_sequence
                    })
                    token_sequence = []
                    tag_sequence = []
        return data

    @staticmethod
    def open_ner_id(path):
        data = []
        token_sequence = []
        tag_sequence = []
        with open(path, "r") as f:
            for line in f:
                if line.strip() != "":
                    words_detail = line.strip().split()
                    token_sequence.append(words_detail[0])
                    tag_sequence.append(
                        words_detail[2] if len(words_detail) == 3 else "O"
                    )
                else:
                    data.append({
                        "tokenized_text": token_sequence,
                        "tag": tag_sequence
                    })
                    token_sequence = []
                    tag_sequence = []
        return data

    @staticmethod
    def open_imdb(directory):
        negative_list = glob(f"{directory}/neg/*.txt")
        positive_list = glob(f"{directory}/pos/*.txt")
        data = []
        for fname in negative_list:
            with open(fname, "r") as f:
                data.append({
                    "text": f.read(),
                    "label": "negative"
                })
        for fname in positive_list:
            with open(fname, "r") as f:
                data.append({
                    "text": f.read(),
                    "label": "positive"
                })
        return pd.DataFrame(data)

    @staticmethod
    def open_news_en(path):
        df = pd.read_csv(path)
        return df.rename({"Class Index": "label", "Title": "text"})

    @staticmethod
    def open_liar_en(path):
        df = pd.read_csv(
            path, delimiter="\t", usecols=[1, 2], header=None,
            names=["label", "text"]
        )
        return df

    @staticmethod
    def open_ner_en(path):
        data = []
        with open(path, "r") as f:
            token_sequence = []
            tag_sequenece = []
            for line in f:
                data_line = line.strip().split("\t")
                if len(data_line) == 2:
                    word, tag = data_line
                    token_sequence.append(word)
                    tag_sequenece.append(tag)
                else:
                    data.append({
                        "tokenized_text": token_sequence,
                        "tag": tag_sequenece
                    })
                    token_sequence = []
                    tag_sequenece = []
        return data


class Preprocess():
    @staticmethod
    def emotion_id(corpus, label):
        preprocessed = tokenization.tokenize(corpus)
        preprocessed = utils.cleaned_corpus(
            preprocessed, stopwords.words('indonesian')
        )
        return utils.split_dataset((preprocessed, label))

    @staticmethod
    def news_category_id(corpus):
        # remove paragraph split
        preprocessed = [s for p in corpus for s in p]
        preprocessed = utils.cleaned_corpus(
            preprocessed, stopwords.words('indonesian')
        )
        return preprocessed

    @staticmethod
    def generic_prepocess(corpus):
        preprocessed = tokenization.tokenize(corpus)
        preprocessed = utils.cleaned_corpus(preprocessed)
        return preprocessed

    def run(corpus):
        return NotImplementedError


TASKS = {
    # https://github.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset
    # Emotion dataset
    "emotion_id": {
        "opener": pd.read_csv,
        "preprocessor": Preprocess.emotion_id
    },
    # https://github.com/kata-ai/indosum
    # News summarization w/ category
    "news_category_id": {
        "opener": Dataset.open_indosum,
        "preprocessor": Preprocess.news_category_id
    },
    # https://github.com/famrashel/idn-tagged-corpus
    # Commonly used indonesian POStagging dataset
    "postag_id": {
        "opener": Dataset.open_postag_id,
        "preprocessor": Preprocess.run
    },
    # https://github.com/UniversalDependencies/UD_Indonesian-GSD
    # Postag on Indonesain UD dataset
    "postag_gsd_id": {
        "opener": Dataset.open_postag_ud,
        "preprocessor": Preprocess.run
    },
    # https://github.com/khairunnisaor/idner-news-2k
    # Indonesian Named Entity Recognition dataset
    "ner_id": {
        "opener": Dataset.open_ner_id,
        "preprocessor": Preprocess.run
    },
    # https://ai.stanford.edu/~amaas/data/sentiment /
    # IMDB dataset
    "sentiment_en": {
        "opener": Dataset.open_imdb,
        "preprocessor": Preprocess.generic_prepocess
    },
    # https://www.kaggle.com/amananandrai/ag-news-classification-dataset
    # Ag News dataset
    "news_category_en": {
        "opener": Dataset.open_news_en,
        "preprocessor": Preprocess.generic_prepocess
    },
    # https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip
    # Liar Dataset
    "fake_news_en": {
        "opener": Dataset.open_liar_en,
        "preprocessor": Preprocess.generic_prepocess
    },
    # https://github.com/UniversalDependencies/UD_English-EWT
    # UD english
    "postag_en": {
        "opener": Dataset.open_postag_ud,
        "preprocessor": Preprocess.run
    },
    # https://github.com/leondz/emerging_entities_17
    # WNUT 2017
    "ner_en": {
        "opener": Dataset.open_ner_en,
        "preprocessor": Preprocess.run
    },
}


def data_opener(path, task):
    return TASKS[task]["opener"](path)


def data_proprocess(corpus, task):
    return TASKS[task]["preprocessor"](corpus)
