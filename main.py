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
    try:
        data = DATASET[args.task](args)
    except KeyError:
        raise KeyError("Invalid Task")

    try:
        DEMOS[args.architecture].main(args, data)
    except KeyError:
        raise KeyError("Invalid sub-command")


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    main(args)
