import pandas as pd
import json

DATA_DIR = "../resources/dataset/"
TASKS = [
    # https://github.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset
    "emotion_id",  # Emotiton dataset
    # https://github.com/kata-ai/indosum
    "news_category_id",  # A news summarization dataset but with category
    # https://github.com/famrashel/idn-tagged-corpus
    "postag_id",  # Postag on commonly used dataset
    # https://github.com/UniversalDependencies/UD_Indonesian-GSD
    "postag_gsd_id",  # Postag on Indonesain UD dataset
    # https://github.com/kxhairunnisaor/idner-news-2k
    "ner_id",  # Named entity Recognition
]


def data_opener(path, task):
    if task == "emotion_id":
        return pd.read_csv(path)
    elif task == "news_category_id":
        return open_indosum(path)
    elif task == "postag_id":
        return open_postag(path)
    elif task == "postag_gsd_id":
        return open_postag_ud(path)
    elif task == "ner_id":
        return open_ner_id(path)


def open_indosum(path):
    data = []
    with open(path, "r") as jsonf:
        for line in jsonf:
            raw_data = json.loads(line)
            data.append({
                "text": raw_data["paragraph"],
                "label": raw_data["category"]
            })
    return pd.DataFrame(data)


def open_postag(path):
    data = []
    with open(path, "r") as f:
        sentences = f.read().split("\n</kalimat>")
        for sentence in sentences:
            token_sequence = []
            tag_sequenece = []
            tokens = sentence.split("\n")
            for idx, token in enumerate(tokens):
                if idx:
                    word, tag = token.split("\t")
                    token_sequence.append(word)
                    tag_sequenece.append(tag)
            data.append({
                "tokenized_text": token_sequence,
                "pos_tag": tag_sequenece
            })
    return data


def open_postag_ud(path):
    data = []
    token_sequence = []
    tag_sequence = []
    with open(path, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            elif line != "":
                words_detail = line.split("\t")
                token_sequence.append(words_detail[1])
                tag_sequence.append(words_detail[3])
            else:
                data.append({
                    "tokenized_text": token_sequence,
                    "pos_tag": tag_sequence
                })
                token_sequence = []
                tag_sequence = []


def open_ner_id(path):
    data = []
    token_sequence = []
    tag_sequence = []
    with open(path, "r") as f:
        for line in f:
            if line != "":
                words_detail = line.split("\t")
                token_sequence.append(words_detail[0])
                tag_sequence.append(words_detail[2])
            else:
                data.append({
                    "tokenized_text": token_sequence,
                    "pos_tag": tag_sequence
                })
                token_sequence = []
                tag_sequence = []
