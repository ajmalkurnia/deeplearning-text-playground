import logging
from sklearn.metrics import classification_report
from model.CNNText.cnn_classifier import CNNClassifier
from keras.optimizers import Adam


def main(args, data):
    logger = logging.getLogger(__name__)
    logger.info("Prepraring data")
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = data.get_data()
    logger.info("Prepraring CNN parameter")
    arch_config = {
        "vocab_size": args.vocabsize,
        "conv_type": args.convtype,
        "embedding_type": args.embeddingtype,
        "embedding_file": args.embeddingfile,
        "input_size": data.get_sequence_length(),
        "optimizer": Adam(lr=0.0001)
    }
    if args.convtype == "parallel":
        arch_config["conv_layers"] = [
            (256, 2, 1, "relu"),
            (256, 3, 1, "relu"),
            (256, 4, 1, "relu"),
            (256, 5, 1, "relu"),
            (256, 6, 1, "relu")
        ]
    else:
        arch_config["conv_layers"] = [
            (256, 5, 3, "relu"),
            # (256, 7, -1, "relu"),
            # (256, 3, -1, "relu"),
            # (256, 3, -1, "relu"),
            (256, 3, -1, "relu"),
            (256, 3, 3, "relu")
        ]
        arch_config["fcn_layers"] = [
            (1024, 0.5, "relu"),
            (1024, 0.5, "relu")
        ]
    if args.loadmodel:
        logger.info("Load model")
        cnn = CNNClassifier.load(args.loadmodel)
    else:
        logger.info("Init model")
        cnn = CNNClassifier(**arch_config)
        logger.info("Training")
        cnn.train(
            X_train, y_train, args.epoch, args.batchsize,
            (X_val, y_val), args.checkpoint
        )
    logger.info("Testing")
    y_pred = cnn.test(X_test)
    # evaluation report
    logger.info("Evaluation")
    logger.info(f"\n{classification_report(y_test, y_pred, digits=5)}")
    if args.savemodel:
        logger.info("Saving file")
        cnn.save(args.savemodel)
