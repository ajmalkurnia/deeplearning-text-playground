import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser = base_args(parser)
    parser = base_embedding_args(parser)
    subparser = parser.add_subparsers(help="Sub-command", dest="architecture")
    subparser.add_parser("rnn", help="Run RNN model")
    subparser.add_parser("cnn", help="Run CNN model")
    subparser.add_parser("transformer", help="Run Transformer model")
    subparser.add_parser("han", help="Run HAN model")
    subparser.add_parser("rcnn", help="Run RCNN model")
    subparser = hybrid_tagger_args(subparser)
    subparser.add_parser("idcnn", help="Run IDCNN model")
    subparser.add_parser("tener", help="Run TENER model")
    return parser


def base_args(parser):
    parser.add_argument(
        "-d", "--datapath", type=str, help="Path to data directory",
        required=True
    )
    parser.add_argument(
        "-t", "--task", type=str, help="Tasks", required=True,
        choices={
            "emotion_id", "news_category_id", "postag_id", "postag_ud_id",
            "ner_id", "sentiment_en", "news_category_en", "fake_news_en",
            "postag_en", "ner_en",
        }
    )
    parser.add_argument(
        "-s", "--savemodel", type=str, help="Path to save model"
    )
    parser.add_argument(
        "-l", "--loadmodel", type=str, help="Path to load model"
    )
    parser.add_argument(
        "--logfile", type=str, help="Log file"
    )
    return parser


def base_embedding_args(parser):
    group = parser.add_argument_group("Pretrained Embedding")
    group.add_argument(
        "--embeddingfile", type=str, help="Pretrained word embedding filepath"
    )
    group.add_argument(
        "--embeddingtype", type=str, choices={"w2v", "ft", "onehot", "custom"},
        help="Word embedding type"
    )
    return parser


def hybrid_tagger_args(subparser):
    hybrid_tag_parser = subparser.add_parser(
        "hybrid", help="Run Hybrid Model"
    )
    hybrid_tag_parser.add_argument(
        "--charlayer", type=str, help="Main Layer settings",
        choices={"cnn", "rnn", "adatrans"}
    )
    hybrid_tag_parser.add_argument(
        "--mainlayer", type=str, help="Main Layer settings",
        choices={"rnn", "adatrans"}, default="rnn", required=True
    )
    hybrid_tag_parser.add_argument(
        "--outlayer", type=str, help="Main Layer settings",
        choices={"crf", "softmax"}, default="crf", required=True
    )
    return subparser
