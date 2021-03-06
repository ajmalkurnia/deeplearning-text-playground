from keras.layers import Embedding, Concatenate, TimeDistributed
from keras.layers import Dropout, Conv1D, Dense

from keras.models import Model, Input
from keras.metrics import Accuracy

from model.base_tagger import BaseTagger


class CNNTagger(BaseTagger):
    def __init__(
        self, embedding_dropout=0.5, pre_outlayer_dropout=0.5,
        conv_layers=[[[128, 3], [128, 5]], [[256, 5]], [[256, 5]]],
        domain_embedding_file=None, domain_embedding_type="glorot_normal",
        domain_embedding_matrix=None, domain_embedding_size=100,
        **kwargs
    ):
        """
        CNN based sequence tagger based on:
        https://www.aclweb.org/anthology/P18-2094/
        Create matrix of custom embedding to use the double embedding input
        the paper is using

        :param embedding_dropout: float, dropout rate after embedding layer
        :param pre_outlayer_dropout: float, dropout rate before output layer
        :param conv_layers: 3D list-like, convolution layer settings,
            each list component denotes config for 1 layer,
                each config consist of 2 length tuple/list that denotes:
                    int, number of filter,
                    int, filter size,
            each convolution will be concatenated for 1 layer,
            each layer is connected sequentially
        """
        super(CNNTagger, self).__init__(**kwargs)
        self.ed = embedding_dropout
        self.pre_outlayer_dropout = pre_outlayer_dropout
        self.conv_layers = conv_layers
        self.domain_embedding_file = domain_embedding_file
        self.domain_embedding_type = domain_embedding_type
        self.domain_embedding_matrix = domain_embedding_matrix
        self.domain_embedding_size = domain_embedding_size

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
        domain_embedding_layer = Embedding(
            self.vocab_size+1, self.domain_embedding_size,
            input_length=self.seq_length,
            embeddings_initializer=self.domain_embedding,
            mask_zero=True,
        )
        domain_embedding_layer = domain_embedding_layer(input_layer)
        embedding_layer = Concatenate()(
            [embedding_layer, domain_embedding_layer]
        )
        embedding_layer = Dropout(self.ed)(embedding_layer)
        x = None
        # CNN layer
        for idx, layer in enumerate(self.conv_layers):
            conv_layers = []
            for filter_num, filter_size in layer:
                conv1 = Conv1D(
                    filter_num, filter_size, activation="relu",
                    padding="same"
                )
                if idx:
                    conv_layers.append(conv1(x))
                else:
                    conv_layers.append(conv1(embedding_layer))
            if len(conv_layers) > 1:
                x = Concatenate(axis=-1)(conv_layers)
            else:
                x = conv_layers[0]
        self.model = Dropout(self.pre_outlayer_dropout)(x)
        # Out dense layer
        out = TimeDistributed(Dense(
            self.n_label+1, activation="softmax"
        ))(self.model)

        self.model = Model(input_layer, out)
        self.model.summary()
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=[Accuracy()]
        )

    def training_prep(self, X, y):
        """
        Initialized necessary class attributes for training

        :param X: 2D list, training dataset in form of tokenized corpus
        :param y: 2D list, training data label
        """
        self.init_inverse_indexes(X, y)
        self.embedding, self.embedding_size = self.init_embedding(
            self.embedding_file, self.embedding_type, self.embedding_size
        )
        domain_embedding = self.init_embedding(
            self.domain_embedding_file, self.domain_embedding_type,
            self.domain_embedding_size
        )
        self.domain_embedding, self.domain_embedding_size = domain_embedding
        self.init_model()

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
        classifier = CNNTagger(**constructor_param)
        classifier.label2idx = class_param["label2idx"]
        classifier.word2idx = class_param["word2idx"]
        classifier.idx2label = class_param["idx2label"]
        classifier.n_label = len(classifier.label2idx)
        return classifier
