from keras.layers import Embedding, Dropout, Conv1D, Dense
from keras.models import Model, Input
from tensorflow_addons.layers.crf import CRF
from keras.metrics import Accuracy

from model.extras.crf_subclass_model import ModelWithCRFLoss
from model.base_crf_out_tagger import BaseCRFTagger


class IDCNNTagger(BaseCRFTagger):
    def __init__(
        self, embedding_dropout=0.5, block_out_dropout=0.5, repeat=1,
        conv_layers=[(300, 3, 1), (300, 3, 2), (300, 3, 1)],
        fcn_layers=[(1024, 0.5, "relu")], **kwargs
    ):
        """
        IDCNN sequence tagger. with CRF as output
        Paper: https://www.aclweb.org/anthology/D17-1283/

        :param embedding_dropout: float, dropout rate after embedding layer
        :param block_out_dropout: float, dropout rate after each idcnn blocks
        :param repeat: int, repeat conv_layers (block) n times
        :param conv_layers: 2D list-like, convolution layer settings,
            each list element denotes config for 1 layer,
                each config consist of 3 length tuple/list that denotes:
                    int, number of filter,
                    int, filter size,
                    int, dilation rate
            each layer is connected sequentially
        :param fcn_layers: 2D list-like, Fully Connected layer settings,
            will be placed after conv layer,
            each list element denotes config for 1 layer,
                each config consist of 3 length tuple/list that denotes:
                    int, number of dense unit
                    float, dropout after dense layer
                    activation, activation function
        """
        super(IDCNNTagger, self).__init__(**kwargs)
        self.ed = embedding_dropout
        self.block_out_dropout = block_out_dropout
        self.conv_layers = conv_layers
        self.fcn_layers = fcn_layers
        self.repeat = repeat
        self.loss = "sparse_categorical_crossentropy"

    def init_model(self):
        """
        Initialize the network model
        """
        # Word Embebedding
        input_layer = Input(shape=(self.seq_length,))
        embedding_layer = Embedding(
            self.vocab_size+1, self.embedding_size,
            input_length=self.seq_length,
            embeddings_initializer=self.embedding,
            mask_zero=True,
        )
        embedding_layer = embedding_layer(input_layer)
        conv_layer = Dropout(self.ed)(embedding_layer)
        # IDCNN Layer
        for idx in range(self.repeat):
            for filter_num, filter_size, d in self.conv_layers:
                conv_layer = Conv1D(
                    filter_num, filter_size,
                    dilation_rate=d,
                    activation="relu",
                    padding="same"
                )(conv_layer)
            conv_layer = Dropout(self.block_out_dropout)(conv_layer)
        # FCN layer
        fcn_layer = conv_layer
        for unit, dropout, activation in self.fcn_layers:
            fcn_layer = Dense(
                unit, activation=activation
            )(fcn_layer)
            fcn_layer = Dropout(dropout)(fcn_layer)
        self.model = Dense(self.n_label+1)(fcn_layer)
        # CRF Layer
        crf = CRF(self.n_label+1)
        out = crf(self.model)
        self.model = Model(inputs=input_layer, outputs=out)
        self.model.summary()
        # Subclassing to properly compute crf loss
        self.model = ModelWithCRFLoss(self.model)
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=[Accuracy()]
        )

    def get_class_param(self):
        class_param = {
            "label2idx": self.label2idx,
            "word2idx": self.word2idx,
            "seq_length": self.seq_length,
            "idx2label": self.idx2label,
        }
        return class_param

    @staticmethod
    def init_from_config(class_param):
        """
        Load model from the saved zipfile

        :param filepath: path to model zip file
        :return classifier: Loaded model class
        """
        constructor_param = {
            "seq_length": class_param["seq_length"]
        }
        classifier = IDCNNTagger(**constructor_param)
        classifier.label2idx = class_param["label2idx"]
        classifier.word2idx = class_param["word2idx"]
        classifier.idx2label = class_param["idx2label"]
        classifier.n_label = len(classifier.label2idx)
        return classifier
