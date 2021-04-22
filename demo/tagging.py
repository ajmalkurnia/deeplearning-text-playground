from model.CNNText.cnn_tagger import CNNTagger
from model.MixedText.hybrid_tagger import DLHybridTagger
from model.CNNText.idcnn_tagger import IDCNNTagger
from model.RNNText.rnn_rnn_tagger import StackedRNNTagger
from model.RNNText.meta_bilstm_tagger import MetaBiLSTMTagger
from model.TransformerText.tener import TENERTagger
from common.configuration import get_config

from sklearn.metrics import classification_report

import logging


def evaluate(pred, ref):
    labels = set([tag for row in ref for tag in row])
    predictions = [tag for row in pred for tag in row]
    truths = [tag for row in ref for tag in row]
    report = classification_report(
        truths, predictions,
        target_names=sorted(list(labels)), digits=4
    )
    return report


TAGGER = {
    "cnn": CNNTagger,
    "hybrid": DLHybridTagger,
    "idcnn": IDCNNTagger,
    "rnn": StackedRNNTagger,
    "tener": TENERTagger,
    "metalstm": MetaBiLSTMTagger
}


def main(args, data):
    logging.getLogger(__name__)

    logging.info("Parse dataset")
    train, test, valid = data.get_data()
    tagger = TAGGER[args.architecture]

    if args.loadmodel:
        logging.info("Load model")
        tagger = tagger.load(args.loadmodel)
    else:
        logging.info(f"{args.architecture} tagger Training")
        class_parameter = get_config(data, args)
        tagger = tagger(**class_parameter)
        tagger.train(
            train[0], train[1], args.epoch, args.batchsize, valid,
            args.checkpoint
        )
    logging.info("Prediction")
    y_pred = tagger.predict(test[0])

    report = evaluate(y_pred, test[1])
    logging.info("Evaluation Results")
    logging.info(f"\n{report}")
    if args.savemodel:
        logging.info("Save model")
        tagger.save(args.savemodel)
