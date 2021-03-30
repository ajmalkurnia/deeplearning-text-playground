import logging
from sklearn.metrics import classification_report
from model.MixedText.rcnn_classifier import RCNNClassifier


def main(args, data):
    logger = logging.getLogger(__name__)
    logger.info("Prepraring data")
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = data.get_data()

    # training, testing
    logger.info("Preparing RCNN parameter")
    arch_config = {
        "vocab_size": args.vocabsize,
        "embedding_type": args.embeddingtype,
        "embedding_file": args.embeddingfile,
        "optimizer": "adam",
        "rnn_size": args.unitrnn,
        "rnn_type": args.typernn,
        "conv_filter": args.convfilter,
        "input_size": data.get_sequence_length()
    }

    if args.loadmodel:
        logger.info("Load model")
        rcnn = RCNNClassifier.load(args.loadmodel)
    else:
        logger.info("Init model")
        rcnn = RCNNClassifier(**arch_config)
        logger.info("Training")
        rcnn.train(
            X_train, y_train, args.epoch, args.batchsize,
            (X_val, y_val), args.checkpoint
        )
    logger.info("Testing")
    y_pred = rcnn.test(X_test)
    # evaluation report
    logger.info(classification_report(y_test, y_pred, digits=5))
    if args.savemodel:
        logger.info("Saving file")
        rcnn.save(args.savemodel)
