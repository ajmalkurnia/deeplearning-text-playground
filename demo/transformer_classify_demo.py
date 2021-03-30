import logging
from sklearn.metrics import classification_report
from model.TransformerText.transformer_classifier import TransformerClassifier


def main(args, data):
    logger = logging.getLogger(__name__)
    logger.info("Prepare dataset")
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = data.get_data()

    # training, testing
    logger.info("Preparing transformer parameter")
    arch_config = {
        "vocab_size": args.vocabsize,
        "embedding_type": args.embeddingtype or "glorot_uniform",
        "embedding_file": args.embeddingfile,
        "optimizer": "adam",
        "dropout": args.dropout,
        "n_blocks": args.nblocks,
        "n_heads": args.nheads,
        "dim_ff": args.dimff,
        "pos_embedding_init": args.positional,
        "input_size": data.get_sequence_length()
    }

    if args.loadmodel:
        logger.info("Load model")
        transformer = TransformerClassifier.load(args.loadmodel)
    else:
        logger.info("Init model")
        transformer = TransformerClassifier(**arch_config)
        logger.info("Training")
        transformer.train(
            X_train, y_train, args.epoch, args.batchsize,
            (X_val, y_val), args.checkpoint
        )
    logger.info("Testing")
    y_pred = transformer.test(X_test)
    # evaluation report
    logger.info("Evaluation")
    logger.info(classification_report(y_test, y_pred))
    if args.savemodel:
        logger.info("Saving file")
        transformer.save(args.savemodel)
