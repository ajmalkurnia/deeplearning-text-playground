from keras.layers import GRU, Embedding, TimeDistributed, Bidirectional
from keras.layers import Dense, LSTM, Dropout
from keras.models import Model, Input

from model.base_classifier import BaseClassifier
from model.base_classifier import Attention
import numpy as np


class HANClassifier(BaseClassifier):
    def __init__(
        self, input_shape=(50, 10), rnn_size=100, dropout=0.2, rnn_type="gru",
        **kwargs
    ):
        """
        Hierarchical Attention Classifier (HAN) constructor,
        Refer to Yang, Z. (2016)
        Beware that the input size of this classifier is a 3D arrays:
            data -> sequence -> subsequence, which can be arranged as:
                data -> token -> char, [[["t", "o", "k", "e", "n"]]], or
                data -> sentence -> token, [[["token", "in", "sentence"]]]
        The original paper used this architecture for document classification
            using second input example
        :param input_shape: tuple (int, int), maximum input shape
            the first element refer to maximum length of a data or
                maximum number of sequence in a data
            the second element refer to maximum length of a sequence or
                maximum number of sub-sequence in a sequence
        :param rnn_size: int, number of rnn hidden units
        :param dropout: float, dropout rate (before softmax)
        :param rnn_type: string, the type of rnn cell, available option:
            gru or lstm
        """
        self.__doc__ = BaseClassifier.__doc__
        kwargs["input_size"] = input_shape[1]
        super(HANClassifier, self).__init__(**kwargs)

        self.max_input_length = input_shape[0]
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.rnn_type = rnn_type

    def init_model(self):
        # Start of lower block (sequence encoder)
        input_layer = Input(shape=(self.max_input, ))
        embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_size,
            input_length=self.max_input,
            embeddings_initializer=self.embedding,
            trainable=self.train_embedding
        )(input_layer)
        lower_rnn_out = self.__get_rnn(embedding_layer)
        lower_attention = Attention("hierarchy")(lower_rnn_out)
        lower_encoder = Model(inputs=input_layer, outputs=lower_attention)
        lower_encoder.summary()

        # Start of higher block (sub-sequence encoder)
        higher_input = Input(shape=(self.max_input_length, self.max_input))
        # run the lower encoder in time distributed fashion
        dist_lower_encoder = TimeDistributed(lower_encoder)(higher_input)
        higher_rnn_out = self.__get_rnn(dist_lower_encoder)
        higher_attention = Attention("hierarchy")(higher_rnn_out)

        # Classifier
        do = Dropout(self.dropout)(higher_attention)
        out = Dense(self.n_label, activation="softmax")(do)
        self.model = Model(higher_input, out)
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"]
        )
        self.model.summary()

    def __get_rnn(self, layer):
        if self.rnn_type == "lstm":
            rnn_out = Bidirectional(
                LSTM(self.rnn_size, return_sequences=True)
            )(layer)
        elif self.rnn_type == "gru":
            rnn_out = Bidirectional(
                GRU(self.rnn_size, return_sequences=True)
            )(layer)
        return rnn_out

    def vectorized_input(self, tokenized_corpus):
        """
        Handling vectorization of 3D tokenized corpus as input since the
            BaseClassifier only handles 2D input
        :param tokenized_corpus: list of list of list of string,
            tokenized corpus with grouped sequence
        :return vector_input: 3D numpy array, indexed sequence of input corpus
        """
        vector_input = np.zeros(
            (len(tokenized_corpus), self.max_input_length, self.max_input),
            dtype=np.int32
        )
        for i, doc in enumerate(tokenized_corpus):
            for j, sentence in enumerate(doc):
                if j == self.max_input_length:
                    break
                for k, token in enumerate(sentence):
                    if k == self.max_input:
                        break
                    vector_input[i][j][k] = self.vocab.get(token, 0)
        return vector_input

    def get_class_param(self):
        return {
            "input_size": self.max_input,
            "l2i": self.label2idx,
            "i2l": self.idx2label,
            "vocab": self.vocab,
            "embedding_size": self.embedding_size,
            "rnn_size": self.rnn_size,
            "dropout": self.dropout,
            "rnn_type": self.rnn_type,
            "max_input_length": self.max_input_length
        }

    @staticmethod
    def get_construtor_param(param):
        return {
            "input_size": param["input_size"],
            "vocab": param["vocab"],
            "embedding_size": param["embedding_size"],
            "input_shape": (param["max_input_length"], param["input_size"]),
            "rnn_size": param["rnn_size"],
            "dropout": param["dropout"],
            "rnn_type": param["rnn_type"]
        }
