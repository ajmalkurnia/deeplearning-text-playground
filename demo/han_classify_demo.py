from sklearn.metrics import classification_report
from model.RNNText.han_classifier import HANClassifier


def main(args, data):

    X_train, y_train, X_test, y_test, X_val, y_val = data
    # Char level
    X_train = [[[*token] for token in doc] for doc in X_train]
    X_test = [[[*token] for token in doc] for doc in X_test]
    X_val = [[[*token] for token in doc] for doc in X_val]
    # Sentence level

    # training, testing
    arch_config = {
        "vocab_size": args.vocabsize,
        "embedding_type": args.embeddingtype,
        "embedding_file": args.embeddingfile,
        "optimizer": "adam",
        "rnn_size": args.unitrnn,
        "dropout": args.dropout,
        "rnn_type": args.typernn
    }

    if args.loadmodel:
        han = HANClassifier.load(args.loadmodel)
    else:
        han = HANClassifier(**arch_config)
        print("Training")
        han.train(
            X_train, y_train, args.epoch, args.batchsize,
            (X_val, y_val), args.checkpoint
        )
    print("Testing")
    y_pred = han.test(X_test)
    # evaluation report
    print(classification_report(y_test, y_pred))
    if args.savemodel:
        print("Saving file")
        han.save(args.savemodel)
