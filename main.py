from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import pandas as pd

from common.tokenization import NLTKToknizerWrapper
from common.util import remove_characters, remove_words
from common.demo_args import get_args
from demo import rnn_classify_demo, cnn_classify_demo
from demo import transformer_classify_demo, han_classify_demo
from demo import rcnn_classify_demo


def main(args):
    print("Open data")
    df = pd.read_csv(args.datapath)

    # tokenize
    print("Tokenize text")
    tokenizer = NLTKToknizerWrapper(False)
    df["token"] = df["text"].apply(lambda t: tokenizer.tokenize(t.lower()))
    # optional preprocessing
    print("Preprocessing")
    id_stopwords = stopwords.words('indonesian')
    cleaned_corpus = []
    for tokens in df["token"]:
        tmp = []
        for token in tokens:
            cleaned_token = remove_characters(token)
            if cleaned_token.strip():
                tmp.append(cleaned_token)
        cleaned_corpus.append(remove_words(tmp, id_stopwords))

    X = cleaned_corpus
    y = df["label"].values.tolist()

    # split
    print("Split Data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=4371
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, train_size=0.9, test_size=0.1, random_state=4371
    )

    data = X_train, y_train, X_test, y_test, X_val, y_val
    if args.architecture == "rnn":
        rnn_classify_demo.main(args, data)
    elif args.architecture == "cnn":
        cnn_classify_demo.main(args, data)
    elif args.architecture == "transformer":
        transformer_classify_demo.main(args, data)
    elif args.architecture == "han":
        han_classify_demo.main(args, data)
    elif args.architecture == "rcnn":
        rcnn_classify_demo.main(args, data)
    else:
        raise ValueError("Invalid sub-command")


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    main(args)
