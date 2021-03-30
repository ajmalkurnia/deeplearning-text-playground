import logging
from sklearn.metrics import classification_report
from model.RNNText.rnn_classifier import RNNClassifier


def main(args, data):
    logger = logging.getLogger(__name__)
    logger.info("Preprare data")
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = data.get_data()

    # training, testing
    logger.info("Preparing RNN Arguments")
    arch_config = {
        "vocab_size": args.vocabsize,
        "embedding_type": args.embeddingtype,
        "embedding_file": args.embeddingfile,
        "optimizer": "adam",
        "rnn_size": args.unitrnn,
        "dropout": args.dropout,
        "rnn_type": args.typernn,
        "attention": args.attention,
        "input_size": data.get_sequence_length()
    }

    if args.loadmodel:
        logger.info("Load model")
        rnn = RNNClassifier.load(args.loadmodel)
    else:
        logger.info("Init model")
        rnn = RNNClassifier(**arch_config)
        logger.info("Training")
        rnn.train(
            X_train, y_train, args.epoch, args.batchsize,
            (X_val, y_val), args.checkpoint
        )
    logger.info("Testing")
    y_pred = rnn.test(X_test)
    # evaluation report
    logger.info("Evaluation")
    logger.info(classification_report(y_test, y_pred, digits=5))
    if args.savemodel:
        logger.info("Saving file")
        rnn.save(args.savemodel)
