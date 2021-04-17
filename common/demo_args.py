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
    subparser = cnn_tagger_args(subparser)
    subparser = idcnn_tagger_args(subparser)
    subparser = tener_tagger_args(subparser)
    subparser = rnn_rnn_tagger_args(subparser)
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
    group1 = hybrid_tag_parser.add_argument_group("Char RNN Settings")
    group1.add_argument(
        "--charrnnunits", type=int, help="RNN unit on char Level", default=25
    )
    group1.add_argument(
        "--charrnndropout", type=float, help="Dropout rate in char-RNN",
        default=0.5
    )
    group2 = hybrid_tag_parser.add_argument_group("Char AdaTrans Settings")
    group2.add_argument(
        "--chartransblocks", type=int, help="Char-transfomer blocks", default=2
    )
    group2.add_argument(
        "--chartransheads", type=int, help="Char-transfomer attention heads",
        default=8
    )
    group2.add_argument(
        "--chartransdimff", type=int, help="Char-transfomer ff dimension",
        default=256
    )
    group2.add_argument(
        "--chartransdropout", type=float, help="Char-transfomer dropout",
        default=0.5
    )
    group2.add_argument(
        "--chartransattdropout", type=float,
        help="Char-transfomer attention dropout", default=0.5
    )
    group2.add_argument(
        "--chartransscale", type=int, help="Char-transfomer attention scaling",
        default=1
    )
    group3 = hybrid_tag_parser.add_argument_group("Main RNN Settings")
    group3.add_argument(
        "--rnnunits", type=int, help="RNN unit on word Level", default=100
    )
    group3.add_argument(
        "--rnndropout", type=float, help="Dropout rate in RNN", default=0.5
    )
    group4 = hybrid_tag_parser.add_argument_group("Main AdaTrans Settings")
    group4.add_argument(
        "--transblocks", type=int, help="Transfomer blocks", default=2
    )
    group4.add_argument(
        "--transheads", type=int, help="Transfomer attention heads",
        default=8
    )
    group4.add_argument(
        "--transdimff", type=int, help="Transfomer ff dimension",
        default=256
    )
    group4.add_argument(
        "--transdropout", type=float, help="Transfomer dropout",
        default=0.5
    )
    group4.add_argument(
        "--transattdropout", type=float,
        help="Transfomer attention dropout", default=0.5
    )
    group4.add_argument(
        "--transscale", type=int, help="Transfomer attention scaling",
        default=1
    )
    group4.add_argument(
        "--transattdim", type=int, help="Transormer attention dimension",
        default=256
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


def tener_tagger_args(subparser):
    tener_tag_parser = subparser.add_parser(
        "tener", help="Run TENER Model"
    )
    tener_tag_parser.add_argument(
        "--nblocks", type=int, help="Transformer block", default=2
    )
    tener_tag_parser.add_argument(
        "--nheads", type=int, help="Attention Heads", default=8
    )
    tener_tag_parser.add_argument(
        "--dimff", type=int, help="Feed forward unit inside transformer",
        default=256
    )
    tener_tag_parser.add_argument(
        "--attentiondim", type=int, help="Unit inside attention",
        default=256
    )
    tener_tag_parser.add_argument(
        "--transformerdropout", type=float, help="Dropout rate in Transformer",
        default=0.5
    )
    tener_tag_parser.add_argument(
        "--attentiondropout", type=float, help="Dropout rate in HeadAttention",
        default=0.5
    )
    tener_tag_parser.add_argument(
        "--embeddingdropout", type=float, help="Dropout rate after embedding",
        default=0.5
    )
    tener_tag_parser.add_argument(
        "--outtransformerdropout", type=float,
        help="Dropout rate after transformer", default=0.5
    )
    return subparser


def rnn_rnn_tagger_args(subparser):
    rnn_tag_parser = subparser.add_parser(
        "rnn-attention", help="Run RNN-Attention-RNN Model"
    )
    rnn_tag_parser.add_argument(
        "--charembedsize", type=int, default=100,
    )
    rnn_tag_parser.add_argument(
        "--charrnnunits", type=int, default=400,
    )
    rnn_tag_parser.add_argument(
        "--rnnunits", type=int, default=400,
    )
    rnn_tag_parser.add_argument(
        "--charrecurrentdropout", type=float, default=0.33,
    )
    rnn_tag_parser.add_argument(
        "--recurrentdropout", type=float, default=0.33,
    )
    rnn_tag_parser.add_argument(
        "--embeddingdropout", type=float, default=0.5,
    )
    rnn_tag_parser.add_argument(
        "--mainlayerdropouts", type=float, default=0.5
    )
    return subparser
