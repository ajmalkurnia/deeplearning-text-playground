from glob import glob

from common.demo_args import get_args
from common.data import open_data, preprocess_data
from common.utils import split_data
from demo import rnn_classify_demo, cnn_classify_demo
from demo import transformer_classify_demo, han_classify_demo
from demo import rcnn_classify_demo


def get_data(args):
    if args.task == "emotion_id":
        df = open_data(args.datapath)
        data = preprocess_data(df["text"].value.to_list(), args.task)
        data = split_data(data, df["label"].value.to_list())
    elif args.task == "news_category_id":
        files = glob(f"{args.datapath}/*.01.jsonl")
        data = [] * 3
        for filen in files:
            df = open_data(filen)
            curr_data = preprocess_data(df["text"], args.task)
            if "dev" in filen:
                data[2] = (curr_data, df["label"].value.to_list())
            elif "test" in filen:
                data[1] = (curr_data, df["label"].value.to_list())
            elif "train" in filen:
                data[0] = (curr_data, df["label"].value.to_list())
    elif args.task == "sentiment_en":
        train_df = open_data(f"{args.datapath}/train/")
        data = preprocess_data(train_df["text"].value.to_list(), args.task)
        train_data, valid_data = split_data(
            [data, train_df["label"].value.to_list()], 0.9, 0.1, 0.0
        )
        test_df = open_data(f"{args.datapath}/test/")
        test_data = (
            preprocess_data(test_df["text"].value.to_list(), args.task),
            test_df["label"].value.to_list()
        )
        data = (train_data, test_data, valid_data)
    elif args.task == "news_category_en":
        train_df = open_data(f"{args.datapath}/train.csv")
        data = preprocess_data(train_df["text"].value.to_list(), args.task)
        train_data, valid_data = split_data(
            [data, train_df["label"].value.to_list()], 0.9, 0.1, 0.0
        )
        test_df = open_data(f"{args.datapath}/test.csv")
        test_data = (
            preprocess_data(test_df["text"].value.to_list(), args.task),
            test_df["label"].value.to_list()
        )
        data = (train_data, test_data, valid_data)
    elif args.task == "fake_news_en":
        data = []
        for i in ["train", "test", "valid"]:
            df = open_data(f"{args.datapath}/{i}.tsv")
            data.append((
                preprocess_data(df["text"].value.to_list(), args.task),
                df["lael"].value.to_list()
            ))
    return data


def main(args):
    # tokenize
    print("Open data")
    data = get_data(args)

    # data = X_train, y_train, X_test, y_test, X_val, y_val
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
