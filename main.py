import logging
from common.demo_args import get_args
from common.data import DATASET
from demo import rnn_classify_demo, cnn_classify_demo
from demo import transformer_classify_demo, han_classify_demo
from demo import rcnn_classify_demo
from demo import cnn_rnn_crf_tag_demo

DEMOS = {
    "rnn-classification": rnn_classify_demo,
    "cnn-classification": cnn_classify_demo,
    "transformer-classification": transformer_classify_demo,
    "han-classification": han_classify_demo,
    "rcnn-classification": rcnn_classify_demo,
    "cnn-rnn-crf-tagger": cnn_rnn_crf_tag_demo
}


def init_log(args):
    log_parameter = {
        "level": logging.INFO,
        "format": "%(levelname)s %(name)s %(funcName)s:%(lineno)d %(message)s"
    }

    if args.logfile:
        log_parameter["filename"] = args.logfile
        log_parameter["filemode"] = "w"
        logging.basicConfig(**log_parameter)
        consolelog = logging.StreamHandler()
        consolelog.setLevel(logging.DEBUG)
        consolelog.setFormatter(
            logging.Formatter(
                "%(levelname)s %(name)s %(funcName)s:%(lineno)d %(message)s"
            )
        )
        logging.getLogger("").addHandler(consolelog)
    else:
        logging.basicConfig(**log_parameter)


def main(args):
    init_log(args)

    try:
        data = DATASET[args.task](args)
    except KeyError:
        raise KeyError("Invalid Task")

    try:
        demo = DEMOS[f"{args.architecture}-{data.TASK}"]
    except KeyError:
        raise KeyError("Invalid sub-command")

    demo.main(args, data)


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    main(args)
