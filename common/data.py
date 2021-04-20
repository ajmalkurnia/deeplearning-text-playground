from nltk.corpus import stopwords
import pandas as pd

import json
from glob import glob
from common import utils
from common import tokenization
from common.utils import split_data


class Dataset():
    def __init__(self, args):
        self.path = args.datapath
        self.arch = args.architecture

    def get_data(self):
        raise NotImplementedError()

    def open_data(self, path):
        raise NotImplementedError()

    def preprocess_data(self, corpus):
        preprocessed = tokenization.tokenize(corpus)
        preprocessed = utils.clean_corpus(preprocessed)
        return preprocessed

    def get_sequence_length(self):
        if self.arch == "han":
            return (20, 10)
        else:
            return 25


class EmotionID(Dataset):
    LANG = "id"
    TASK = "classification"

    def __init__(self, args):
        super(EmotionID, self).__init__(args)

    def open_data(self, path):
        return pd.read_csv(path)

    def preprocess_data(self, corpus):
        preprocessed = tokenization.tokenize(corpus)
        preprocessed = utils.clean_corpus(preprocessed)
        return preprocessed

    def preprocess_nested_data(self, corpus):
        preprocessed = tokenization.tokenize(corpus, "nltk_sentence")
        preprocessed_data = []
        for data in preprocessed:
            tokenized_corpus = tokenization.tokenize(data)
            preprocessed = utils.clean_corpus(
                tokenized_corpus
            )
            preprocessed_data.append(preprocessed)
        return preprocessed_data

    def get_data(self):
        df = self.open_data(f"{self.path}/Twitter_Emotion_Dataset.csv")
        data = self.preprocess_data(df["text"])
        data = split_data((data, df["label"]))
        return data

    def get_sequence_length(self):
        if self.arch == "han":
            return (50, 10)
        else:
            return 50


class IndoSum(Dataset):
    LANG = "id"
    TASK = "classification"

    def __init__(self, args):
        super(IndoSum, self).__init__(args)

    def open_data(self, path):
        data = []
        with open(path, "r") as jsonf:
            for line in jsonf:
                raw_data = json.loads(line)
                data.append({
                    "text": raw_data["paragraphs"],
                    "label": raw_data["category"]
                })
        return pd.DataFrame(data)

    def flatten_corpus(self, corpus):
        flat_data = []
        counter_sentence = 0
        counter_token = 0
        for data in corpus.values.tolist():
            tmp = []
            for paragraph in data:
                for sentence in paragraph:
                    counter_sentence += 1
                    for token in sentence:
                        counter_token += 1
                        tmp.append(token)
            flat_data.append(tmp)
        return flat_data

    def preprocess_data(self, corpus):
        preprocessed = self.flatten_corpus(corpus)
        preprocessed = utils.clean_corpus(
            preprocessed, stopwords.words('indonesian')
        )
        return preprocessed

    def preprocess_nested_data(self, corpus):
        preprocessed = []
        for data in corpus.values.tolist():
            tmp = []
            for paragraph in data:
                for sentence in paragraph:
                    tmp.append(sentence)
            preprocessed.append(
                utils.clean_corpus(tmp, stopwords.words('indonesian'))
            )
        return preprocessed

    def get_data(self):
        files = glob(f"{self.path}/*.01.jsonl")
        data = [None] * 3
        for filen in files:
            df = self.open_data(filen)
            if self.arch == "han":
                curr_data = self.preprocess_nested_data(df["text"])
            else:
                curr_data = self.preprocess_data(df["text"])
            if "dev" in filen:
                data[2] = (curr_data, df["label"])
            elif "test" in filen:
                data[1] = (curr_data, df["label"])
            elif "train" in filen:
                data[0] = (curr_data, df["label"])
        return data

    def get_sequence_length(self):
        if self.arch == "han":
            return (20, 20)
        else:
            return 250


class POSTagID(Dataset):
    LANG = "id"
    TASK = "tagger"

    def open_data(self, path):
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

    def get_data(self):
        path = f"{self.path}/Indonesian_Manually_Tagged_Corpus_ID.tsv"
        df = self.open_data(path)
        X = [row["tokenized_text"] for row in df]
        y = [row["tag"] for row in df]

        data = split_data((X, y))
        return data

    def get_sequence_length(self):
        return 90


class POSTagUDID(Dataset):
    LANG = "id"
    TASK = "tagger"

    def __init__(self, args):
        super(POSTagUDID, self).__init__(args)

    def open_data(self, path):
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

    def get_data(self):
        files = glob(f"{self.path}/*.conllu")
        data = [None] * 3
        for filen in files:
            df = self.open_data(filen)
            X = [row["tokenized_text"] for row in df]
            y = [row["tag"] for row in df]
            if "dev" in filen:
                data[2] = (X, y)
            elif "test" in filen:
                data[1] = (X, y)
            elif "train" in filen:
                data[0] = (X, y)
        return data

    def get_sequence_length(self):
        return 190


class NERID(Dataset):
    LANG = "id"
    TASK = "tagger"

    def __init__(self, args):
        super(NERID, self).__init__(args)

    def open_data(self, path):
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

    def get_data(self):
        files = glob(f"{self.path}/*.txt")
        data = [None] * 3
        for filen in files:
            df = self.open_data(filen)
            X = [row["tokenized_text"] for row in df]
            y = [row["tag"] for row in df]
            if "dev" in filen:
                data[2] = (X, y)
            elif "test" in filen:
                data[1] = (X, y)
            elif "train" in filen:
                data[0] = (X, y)
        return data

    def get_sequence_length(self):
        return 75


class IMDB(Dataset):
    LANG = "en"
    TASK = "classification"

    def __init__(self, args):
        super(IMDB, self).__init__(args)

    def open_data(self, path):
        negative_list = glob(f"{path}/neg/*.txt")
        positive_list = glob(f"{path}/pos/*.txt")
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

    def get_data(self):
        train_df = self.open_data(f"{self.path}/train/")
        if self.arch == "han":
            preprocessor = self.preprocess_nested_data
        else:
            preprocessor = self.preprocess_data
        data = preprocessor(train_df["text"])
        train_data, valid_data, _ = split_data(
            (data, train_df["label"]), 90, 10, 0
        )
        test_df = self.open_data(f"{self.path}/test/")
        test_data = (
            preprocessor(test_df["text"]),
            test_df["label"]
        )
        data = (train_data, test_data, valid_data)
        return data

    def get_sequence_length(self):
        if self.arch == "han":
            return (20, 10)
        else:
            return 200


class AGNews(Dataset):
    LANG = "en"
    TASK = "classification"

    def __init__(self, args):
        super(AGNews, self).__init__(args)

    def open_data(self, path):
        df = pd.read_csv(path)
        return df.rename(columns={
            "Class Index": "label",
            "Description": "text"
            })

    def get_data(self):
        train_df = self.open_data(f"{self.path}/train.csv")
        data = self.preprocess_data(train_df["text"])
        train_data, valid_data, _ = split_data(
            (data, train_df["label"]), 90, 10, 0
        )
        test_df = self.open_data(f"{self.path}/test.csv")
        test_data = (
            self.preprocess_data(test_df["text"]),
            test_df["label"]
        )
        data = (train_data, test_data, valid_data)
        return data

    def get_sequence_length(self):
        if self.arch == "han":
            return (20, 10)
        else:
            return 50


class LIAR(Dataset):
    LANG = "en"
    TASK = "classification"

    def __init__(self, args):
        super(LIAR, self).__init__(args)

    def open_data(self, path):
        df = pd.read_csv(
            path, delimiter="\t", usecols=[1, 2], header=None,
            names=["label", "text"]
        )
        return df

    def get_data(self):
        data = []
        for i in ["train", "test", "valid"]:
            df = self.open_data(f"{self.path}/{i}.tsv")
            data.append((
                self.preprocess_data(df["text"]),
                df["label"]
            ))
        return data

    def get_sequence_length(self):
        if self.arch == "han":
            return (20, 10)
        else:
            return 25


class NEREN(Dataset):
    LANG = "en"
    TASK = "tagger"

    def __init__(self, args):
        super(NEREN, self).__init__(args)

    def open_data(self, path):
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

    def get_data(self):
        files = [
            f"{self.path}/wnut17train.conll",
            f"{self.path}/emerging.dev.conll",
            f"{self.path}/emerging.test.annotated",
        ]
        data = [None] * 3
        for filen in files:
            df = self.open_data(filen)
            X = [row["tokenized_text"] for row in df]
            y = [row["tag"] for row in df]
            if "dev" in filen:
                data[2] = (X, y)
            elif "test" in filen:
                data[1] = (X, y)
            elif "train" in filen:
                data[0] = (X, y)
        return data

    def get_sequence_length(self):
        return 105


DATASET = {
    # https://github.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset
    # Emotion dataset
    "emotion_id": EmotionID,
    # https://github.com/kata-ai/indosum
    # News summarization w/ category
    "news_category_id": IndoSum,
    # https://github.com/famrashel/idn-tagged-corpus
    # Commonly used indonesian POStagging dataset
    "postag_id": POSTagID,
    # https://github.com/UniversalDependencies/UD_Indonesian-GSD
    # Postag on Indonesain UD dataset
    "postag_ud_id": POSTagUDID,
    # https://github.com/khairunnisaor/idner-news-2k
    # Indonesian Named Entity Recognition dataset
    "ner_id": NERID,
    # https://ai.stanford.edu/~amaas/data/sentiment /
    # IMDB dataset
    "sentiment_en": IMDB,
    # https://www.kaggle.com/amananandrai/ag-news-classification-dataset
    # Ag News dataset
    "news_category_en": AGNews,
    # https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip
    # Liar Dataset
    "fake_news_en": LIAR,
    # https://github.com/UniversalDependencies/UD_English-EWT
    # UD english
    "postag_en": POSTagUDID,  # Its the same format so just re-use this
    # https://github.com/leondz/emerging_entities_17
    # WNUT 2017
    "ner_en": NEREN
}
