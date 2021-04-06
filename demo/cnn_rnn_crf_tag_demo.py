from model.MixedText.hybrid_tagger import DLHybridTagger
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
        hybrid_tagger = DLHybridTagger.load(args.loadmodel)
    else:
        logging.info("DL Hybrid tagger Training")
        class_parameter = {
            "word_embed_file": args.embeddingfile,
            "we_type": args.embeddingtype,
            "seq_length": data.get_sequence_length(),
            "use_crf": True,
            "use_cnn": True
        }

        hybrid_tagger = DLHybridTagger(**class_parameter)
        hybrid_tagger.train(
            train[0], train[1], args.epoch, valid
        )
    logging.info("Prediction")
    y_pred = hybrid_tagger.predict(test[0])

    report = evaluate(y_pred, test[1])
    logging.info("Evaluation Results")
    logging.info(f"\n{report}")
    if args.savemodel:
        logging.info("Save model")
        hybrid_tagger.save(args.savemodel)
