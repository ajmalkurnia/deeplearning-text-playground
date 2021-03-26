import pandas as pd
import json
from glob import glob

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


TASKS_OPENER = {
    # https://github.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset
    "emotion_id": pd.read_csv,  # Emotion dataset
    # https://github.com/kata-ai/indosum
    "news_category_id": Dataset.open_indosum,  # News summarization w/ category
    # https://github.com/famrashel/idn-tagged-corpus
    "postag_id": Dataset.open_postag_id,  # Postag on commonly used dataset
    # https://github.com/UniversalDependencies/UD_Indonesian-GSD
    "postag_gsd_id": Dataset.open_postag_ud,  # Postag on Indonesain UD dataset
    # https://github.com/khairunnisaor/idner-news-2k
    "ner_id": Dataset.open_ner_id,  # Named entity Recognition
    # https: // ai.stanford.edu/~amaas/data/sentiment /
    "sentiment_en": Dataset.open_imdb,  # IMDB dataset
    # https: // www.kaggle.com/amananandrai/ag-news-classification-dataset
    "news_category_en": Dataset.open_news_en,   # Ag News
    # https: // sites.cs.ucsb.edu/~william/data/liar_dataset.zip
    "fake_news_en": Dataset.open_liar_en,  # Liar Dataset
    # https: // github.com/UniversalDependencies/UD_English-EWT
    "postag_en": Dataset.open_postag_ud,  # UD english
    # https: // github.com/leondz/emerging_entities_17
    "ner_en": Dataset.open_ner_en,  # WNUT 2017
}


def data_opener(path, task):
    return TASKS_OPENER[task](path)
