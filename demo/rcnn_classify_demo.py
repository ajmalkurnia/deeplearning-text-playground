from sklearn.metrics import classification_report
from model.MixedText.rcnn_classifier import RCNNClassifier


def main(args, data):

    (X_train, y_train), (X_test, y_test), (X_val, y_val) = data

    # training, testing
    arch_config = {
        "vocab_size": args.vocabsize,
        "embedding_type": args.embeddingtype,
        "embedding_file": args.embeddingfile,
        "optimizer": "adam",
        "rnn_size": args.unitrnn,
        "rnn_type": args.typernn,
        "conv_filter": args.convfilter
    }

    if args.loadmodel:
        rcnn = RCNNClassifier.load(args.loadmodel)
    else:
        rcnn = RCNNClassifier(**arch_config)
        print("Training")
        rcnn.train(
            X_train, y_train, args.epoch, args.batchsize,
            (X_val, y_val), args.checkpoint
        )
    print("Testing")
    y_pred = rcnn.test(X_test)
    # evaluation report
    print(classification_report(y_test, y_pred))
    if args.savemodel:
        print("Saving file")
        rcnn.save(args.savemodel)
