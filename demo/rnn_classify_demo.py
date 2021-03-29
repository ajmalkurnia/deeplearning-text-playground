from sklearn.metrics import classification_report
from model.RNNText.rnn_classifier import RNNClassifier


def main(args, data):

    (X_train, y_train), (X_test, y_test), (X_val, y_val) = data.get_data()

    # training, testing
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
        rnn = RNNClassifier.load(args.loadmodel)
    else:
        rnn = RNNClassifier(**arch_config)
        print("Training")
        rnn.train(
            X_train, y_train, args.epoch, args.batchsize,
            (X_val, y_val), args.checkpoint
        )
    print("Testing")
    y_pred = rnn.test(X_test)
    # evaluation report
    print(classification_report(y_test, y_pred))
    if args.savemodel:
        print("Saving file")
        rnn.save(args.savemodel)
