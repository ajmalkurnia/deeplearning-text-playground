from sklearn.metrics import classification_report
from model.TransformerText.transformer_classifier import TransformerClassifier


def main(args, data):

    X_train, y_train, X_test, y_test, X_val, y_val = data

    # training, testing
    arch_config = {
        "vocab_size": args.vocabsize,
        "embedding_type": args.embeddingtype or "glorot_uniform",
        "embedding_file": args.embeddingfile,
        "optimizer": "adam",
        "dropout": args.dropout,
        "n_blocks": args.nblocks,
        "n_heads": args.nheads,
        "dim_ff": args.dimff,
        "pos_embedding_init": args.positional
    }

    if args.loadmodel:
        transformer = TransformerClassifier()
        transformer.load(args.loadmodel)
    else:
        transformer = TransformerClassifier(**arch_config)
        print("Training")
        transformer.train(
            X_train, y_train, args.epoch, args.batchsize,
            (X_val, y_val), args.checkpoint
        )
    print("Testing")
    y_pred = transformer.test(X_test)
    # evaluation report
    print(classification_report(y_test, y_pred))
    if args.savemodel:
        print("Saving file")
        transformer.save(args.savemodel)
