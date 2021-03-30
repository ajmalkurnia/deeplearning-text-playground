import logging
from common.demo_args import get_args
from common.data import DATASET
from demo import rnn_classify_demo, cnn_classify_demo
from demo import transformer_classify_demo, han_classify_demo
from demo import rcnn_classify_demo
DEMOS = {
    "rnn": rnn_classify_demo,
    "cnn": cnn_classify_demo,
    "transformer": transformer_classify_demo,
    "han": han_classify_demo,
    "rcnn": rcnn_classify_demo
}


def main(args):
    log_parameter = {
        "level": logging.INFO,
        "format": "%(levelname)s %(name)s %(funcname)s:%(lineno)d %(message)s"
    }
    if args.logfile:
        log_parameter["filename"] = args.logfile
        log_parameter["filemode"] = "w"

    logging.basicConfig(**log_parameter)

    try:
        data = DATASET[args.task](args)
    except KeyError:
        raise KeyError("Invalid Task")

    try:
        demo = DEMOS[args.architecture]
    except KeyError:
        raise KeyError("Invalid sub-command")

    demo.main(args, data)


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    main(args)
