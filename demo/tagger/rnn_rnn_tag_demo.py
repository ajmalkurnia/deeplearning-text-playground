from model.RNNText.rnn_rnn_tagger import StackedRNNTagger
from sklearn.metrics import classification_report
import logging


def evaluate(pred, ref):
    labels = set([tag for row in ref for tag in row])
    predictions = [tag for row in pred for tag in row]
    truths = [tag for row in ref for tag in row]
    report = classification_report(
        truths, predictions,
        target_names=sorted(list(labels)), digits=4
    )
    return report


def main(args, data):
    logging.getLogger(__name__)

    logging.info("Parse dataset")
    train, test, valid = data.get_data()

    if args.loadmodel:
        logging.info("Load model")
        hybrid_tagger = StackedRNNTagger.load(args.loadmodel)
    else:
        logging.info("DL Hybrid tagger Training")
        class_parameter = {
            "embedding_file": args.embeddingfile,
            "embedding_type": args.embeddingtype,
            "seq_length": data.get_sequence_length(),
            "char_embed_size": args.charembedsize,
            "char_rnn_units": args.charrnnunits,
            "char_recurrent_dropout": args.charrecurrentdropout,
            "recurrent_dropout": args.recurrentdropout,
            "rnn_units": args.rnnunits,
            "embedding_dropout": args.embeddingdropout,
            "main_layer_dropout": args.mainlayerdropouts
        }

        hybrid_tagger = StackedRNNTagger(**class_parameter)
        hybrid_tagger.train(
            train[0], train[1], args.epoch, valid, args.batchsize
        )
    logging.info("Prediction")
    y_pred = hybrid_tagger.predict(test[0])

    report = evaluate(y_pred, test[1])
    logging.info("Evaluation Results")
    logging.info(f"\n{report}")
    if args.savemodel:
        logging.info("Save model")
        hybrid_tagger.save(args.savemodel)
