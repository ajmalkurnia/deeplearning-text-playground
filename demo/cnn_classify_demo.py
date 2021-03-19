from sklearn.metrics import classification_report
from model.CNNText.cnn_classifier import CNNClassifier


def main(args, data):

    X_train, y_train, X_test, y_test, X_val, y_val = data
    # training, testing
    arch_config = {
        "vocab_size": args.vocabsize,
        "conv_type": args.convtype,
        "embedding_type": args.embeddingtype,
        "embedding_file": args.embeddingfile,
        "optimizer": "adagrad"
    }
    if args.convtype == "parallel":
        arch_config["conv_layers"] = [
            (256, 3, 1, "relu"),
            (256, 4, 1, "relu"),
            (256, 5, 1, "relu")
        ]
    else:
        arch_config["conv_layers"] = [
            (256, 7, 3, "relu"),
            # (256, 7, -1, "relu"),
            # (256, 3, -1, "relu"),
            # (256, 3, -1, "relu"),
            # (256, 3, -1, "relu"),
            (256, 3, 3, "relu")
        ]
        arch_config["fcn_layers"] = [
            (1024, 0.5, "relu"),
            (1024, 0.5, "relu")
        ]
    if args.loadmodel:
        cnn = CNNClassifier()
        cnn.load(args.loadmodel)
    else:
        cnn = CNNClassifier(**arch_config)
        print("Training")
        cnn.train(
            X_train, y_train, args.epoch, args.batchsize,
            (X_val, y_val), args.checkpoint
        )
    print("Testing")
    y_pred = cnn.test(X_test)
    # evaluation report
    print(classification_report(y_test, y_pred))
    if args.savemodel:
        print("Saving file")
        cnn.save(args.savemodel)
