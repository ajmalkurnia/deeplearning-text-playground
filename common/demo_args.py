import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser = base_args(parser)
    parser = base_embedding_args(parser)
    parser = base_dl_args(parser)
    subparser = parser.add_subparsers(help="Sub-command", dest="architecture")
    subparser = rnn_args(subparser)
    subparser = cnn_args(subparser)
    subparser = transformer_args(subparser)
    subparser = han_args(subparser)
    subparser = rcnn_args(subparser)
    subparser = hybrid_tagger_args(subparser)
    subparser = hybrid_tagger_args(subparser, "cnn-rnn")
    subparser = hybrid_tagger_args(subparser, "rnn-crf")
    subparser = hybrid_tagger_args(subparser, "rnn-seq")
    subparser = cnn_tagger_args(subparser)
    subparser = idcnn_tagger_args(subparser)
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


def base_dl_args(parser):
    group = parser.add_argument_group("Base deep learning command")
    group.add_argument(
        "-v", "--vocabsize", type=int, help="Vocabulary size", default=3000
    )
    group.add_argument(
        "-e", "--epoch", type=int, help="Epoch Training", default=10
    )
    group.add_argument(
        "-b", "--batchsize", type=int, help="Number of batch", default=32
    )
    group.add_argument(
        "-c", "--checkpoint", type=str, help="Checkpoint path"
    )
    return parser


def rnn_args(subparser):
    # subparser = parser.add_subparsers(help="RNN Model")
    rnn_parser = subparser.add_parser("rnn", help="Run RNN model")
    rnn_parser.add_argument(
        "-u", "--unitrnn", type=int, help="RNN unit size", default=100
    )
    rnn_parser.add_argument(
        "--typernn", type=str, help="Type of RNN layer",
        choices={"lstm", "gru"}, default="lstm"
    )
    rnn_parser.add_argument(
        "--dropout", type=float, help="Droupout value", default=0.5
    )
    rnn_parser.add_argument(
        "-a", "--attention", type=str, help="Attention type",
        choices={"self", "add", "dot", "scale", "general", "location"})
    return subparser


def cnn_args(subparser):
    cnn_parser = subparser.add_parser("cnn", help="Run CNN model")
    cnn_parser.add_argument(
        "--convtype", type=str, choices={"parallel", "sequential"},
        help="Convolution process type", default="sequence"
    )
    return subparser


def transformer_args(subparser):
    transformer_parser = subparser.add_parser(
        "transformer", help="Run Transformer Model"
    )
    transformer_parser.add_argument(
        "--nblocks", type=int, help="Number of transformer blocks",
        default=2
    )
    transformer_parser.add_argument(
        "--nheads", type=int, help="Number of attention heads",
        default=6
    )
    transformer_parser.add_argument(
        "--dimff", type=int, help="Feed forward unit inside transformer",
        default=256
    )
    transformer_parser.add_argument(
        "--dropout", type=float, help="Dropout rate",
        default=0.3
    )
    transformer_parser.add_argument(
        "--positional", action="store_true",
        help="Initialized positinal embedding with sincos"
    )
    return subparser


def han_args(subparser):
    han_parser = subparser.add_parser(
        "han", help="Run HAN Model"
    )
    han_parser.add_argument(
        "-u", "--unitrnn", type=int, help="RNN unit size", default=100
    )
    han_parser.add_argument(
        "--typernn", type=str, help="Type of RNN layer",
        choices={"lstm", "gru"}, default="lstm"
    )
    han_parser.add_argument(
        "--dropout", type=float, help="Droupout value", default=0.2
    )
    return subparser


def rcnn_args(subparser):
    rcnn_parser = subparser.add_parser(
        "rcnn", help="Run RCNN Model"
    )
    rcnn_parser.add_argument(
        "-u", "--unitrnn", type=int, help="RNN unit size", default=100
    )
    rcnn_parser.add_argument(
        "--typernn", type=str, help="Type of RNN layer",
        choices={"lstm", "gru"}, default="lstm"
    )
    rcnn_parser.add_argument(
        "--convfilter", type=int, help="Number of convolution filter",
        default=128
    )
    return subparser


def hybrid_tagger_args(subparser, name="cnn-rnn-crf"):
    hybrid_tag_parser = subparser.add_parser(
        name, help=f"Run {name.upper()} Model"
    )
    hybrid_tag_parser.add_argument(
        "-u", "--unitrnn", type=int, help="RNN unit size", default=100
    )
    hybrid_tag_parser.add_argument(
        "--charembedsize", type=int, help="Character embedding size",
        default=30
    )
    hybrid_tag_parser.add_argument(
        "--recurrentdropout", type=float, help="Dropout rate inside RNN",
        default=0.5
    )
    hybrid_tag_parser.add_argument(
        "--embeddingdropout", type=float, help="Dropout rate after embedding",
        default=0.5
    )
    hybrid_tag_parser.add_argument(
        "--preoutputdropout", type=float, help="Dropout rate before output",
        default=0.5
    )
    return subparser


def cnn_tagger_args(subparser):
    cnn_sub_parser = subparser.add_parser(
        "cnn-seq", help="Run CNN Model"
    )
    cnn_sub_parser.add_argument(
        "--embeddingdropout", type=float, help="Dropout rate after embedding",
        default=0.5
    )
    cnn_sub_parser.add_argument(
        "--preoutputdropout", type=float, help="Dropout rate before output",
        default=0.5
    )
    return subparser


def idcnn_tagger_args(subparser):
    idcnn_subparser = subparser.add_parser(
        "idcnn", help="Run CNN Model"
    )
    idcnn_subparser.add_argument(
        "--embeddingdropout", type=float, help="Dropout rate after embedding",
        default=0.5
    )
    idcnn_subparser.add_argument(
        "--blockdropout", type=float, help="Dropout rate at the end of blocks",
        default=0.5
    )
    idcnn_subparser.add_argument(
        "--repeat", type=int, help="Repeat block x times",
        default=1
    )
    return subparser
