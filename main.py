import logging
from common.demo_args import get_args
from common.data import DATASET
from demo import classify
from demo import tagging

DEMOS = {
    "classification": classify,
    "tagger": tagging,
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
        demo = DEMOS[data.TASK]
    except KeyError:
        raise KeyError("Invalid dataset")

    demo.main(args, data)


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    main(args)
