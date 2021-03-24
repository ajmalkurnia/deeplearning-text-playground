from keras.layers import Input, LSTM, GRU, Concatenate, Conv1D, Dropout, Dense
from keras.layers import Embedding, Lambda, GlobalMaxPooling1D
from keras.models import Model
from keras import backend as K
from model.base_classifier import BaseClassifier


class RCNNClassifier(BaseClassifier):
    def __init__(self, rnn_size, rnn_type, conv_filter, fcn_layer, **kwargs):
        """
        Class constructor
        :param rnn_size: int, the size of rnn units
        :param rnn_type: string, rnn cell type to be used "lstm"/"gru"
        :param conv_filter: int, number of filter on convolution layer
        :param fcn_layer: list of tuple, configuration of fcn layer,
            after convolution, each tuple is after consist of:
                [int] number of units,
                [float] dropout after fcn layer,
                [string] activation function,
        """
        self.__doc__ = self.__doc__ + super().__doc__
        super(RCNNClassifier, self).__init__(**kwargs)
        self.rnn_size = rnn_size
        self.rnn_type = rnn_type
        self.fcn_layer = fcn_layer
        self.conv_filter = conv_filter

    def init_model(self):
        input_layer = Input(shape=(self.max_input, ))
        embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_size,
            input_length=self.max_input,
            embeddings_initializer=self.embedding,
            trainable=self.train_embedding
        )
        center_embedding = embedding_layer(input_layer)
        # Get left context by shifting the data to the right
        left_context = Lambda(
            lambda center: Concatenate(axis=1)([
                K.zeros((center.shape[0], 1, center.shape[1])), center[:, :-1]
            ])
        )(center_embedding)
        left_context_rnn = self.__get_rnn()(
            self.units, return_sequences=True
        )(left_context)
        # Get right context by shifting the data to the left
        right_context = Lambda(
            lambda center: Concatenate(axis=1)([
                center[:, 1:], K.zeros((center.shape[0], 1, center.shape[1]))
            ])
        )(center_embedding)
        right_context_rnn = self.__get_rnn()(
            self.units, return_sequences=True, go_backwards=True
        )(right_context)
        # Right context processed from last to first
        right_context_rnn = Lambda(
            lambda right: K.reverse(right, axes=1)
        )(right_context_rnn)
        # Convolution Part
        self.model = Concatenate(axis=-1)([
            left_context_rnn, center_embedding, right_context_rnn
        ])
        self.model = Conv1D(
            self.conv_filter, 1, activation="tanh"
        )(self.model)
        self.model = GlobalMaxPooling1D()(self.model)
        # FCN for classification
        for units, do_rate, activation in self.fcn_layers:
            self.model = Dense(units, activation=activation)(self.model)
            self.model = Dropout(do_rate)(self.model)
        output = Dense(self.n_label, "softmax")(self.model)
        self.model = Model(input_layer, output)
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"]
        )
        self.model.summary()

    def __get_rnn(self):
        if self.rnn_type == "lstm":
            return LSTM
        elif self.rnn_type == "gru":
            return GRU

    def get_class_param(self):
        return {
            "input_size": self.max_input,
            "l2i": self.label2idx,
            "i2l": self.idx2label,
            "vocab": self.vocab,
            "embedding_size": self.embedding_size,
            "rnn_size": self.rnn_size,
            "rnn_type": self.rnn_type,
            "fcn_layer": self.fcn_layer,
            "conv_filter": self.conv_filter
        }

    @staticmethod
    def get_construtor_param(param):
        return {
            "input_size": param["max_input"],
            "vocab": param["vocab"],
            "embedding_size": param["embedding_size"],
            "rnn_size": param["rnn_size"],
            "rnn_type": param["rnn_type"],
            "fcn_layer": param["fcn_layer"],
            "conv_filter": param["conv_filter"]
        }
