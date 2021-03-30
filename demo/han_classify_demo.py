import logging
from sklearn.metrics import classification_report
from model.RNNText.han_classifier import HANClassifier


def main(args, data):
    logger = logging.getLogger(__name__)
    logger.info("Prepraring data")
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = data.get_data()
    # Char level
    logger.info("Converting 2D data to 3D")

    X_train = [[[*token] for token in doc] for doc in X_train]
    X_test = [[[*token] for token in doc] for doc in X_test]
    X_val = [[[*token] for token in doc] for doc in X_val]
    # Sentence level

    # training, testing
    logger.info("Preparing HAN parameter")
    arch_config = {
        "vocab_size": args.vocabsize,
        "embedding_type": args.embeddingtype,
        "embedding_file": args.embeddingfile,
        "optimizer": "adam",
        "rnn_size": args.unitrnn,
        "dropout": args.dropout,
        "rnn_type": args.typernn,
        "input_shape": data.get_sequence_length()
    }

    if args.loadmodel:
        logger.info("Load model")
        han = HANClassifier.load(args.loadmodel)
    else:
        logger.info("Init model")
        han = HANClassifier(**arch_config)
        logger.info("Training")
        han.train(
            X_train, y_train, args.epoch, args.batchsize,
            (X_val, y_val), args.checkpoint
        )
    logger.info("Testing")
    y_pred = han.test(X_test)
    # evaluation report
    logger.info("Evaluation")
    logger.info(classification_report(y_test, y_pred))
    if args.savemodel:
        logger.info("Saving file")
        han.save(args.savemodel)
