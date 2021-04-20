from model.CNNText.cnn_classifier import CNNClassifier
from model.RNNText.han_classifier import HANClassifier
from model.RNNText.rnn_classifier import RNNClassifier
from model.MixedText.rcnn_classifier import RCNNClassifier
from model.TransformerText.transformer_classifier import TransformerClassifier
from common.configuration import get_config

from sklearn.metrics import classification_report

import logging


CLASSIFIER = {
    "cnn": CNNClassifier,
    "han": HANClassifier,
    "rnn": RNNClassifier,
    "rcnn": RCNNClassifier,
    "transformer": TransformerClassifier
}


def main(args, data):
    logger = logging.getLogger(__name__)
    logger.info("Prepraring data")
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = data.get_data()
    logger.info(f"Prepraring {args.architecture} parameter")
    arch_config = {}

    classifier = CLASSIFIER[args.architecture]
    if args.loadmodel:
        logger.info("Load model")
        classifier = classifier.load(args.loadmodel)
    else:
        logger.info("Init model")
        arch_config = get_config(data, args)
        classifier = classifier(**arch_config)
        logger.info("Training")
        classifier.train(
            X_train, y_train, args.epoch, args.batchsize,
            (X_val, y_val), args.checkpoint
        )
    logger.info("Testing")
    y_pred = classifier.test(X_test)
    # evaluation report
    logger.info("Evaluation")
    logger.info(f"\n{classification_report(y_test, y_pred, digits=5)}")
    if args.savemodel:
        logger.info("Saving file")
        classifier.save(args.savemodel)
