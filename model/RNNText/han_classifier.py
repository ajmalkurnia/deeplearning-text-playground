from keras.layers import GRU, Embedding, TimeDistributed, Bidirectional
from keras.layers import Dense, Concatenate, LSTM
from keras.models import Model, Input, Dropout

from model.base_classifier import BaseClassifier
from model.base_classifier import Attention
import numpy as np


class HANClassifier(BaseClassifier):
    def __init__(
        self, input_shape=(100, 30), rnn_size=100, lowest_level="char",
        dropout=0.2, rnn_type="gru",
        **kwargs
    ):
        kwargs["max_input"] = input_shape[1]
        super(HANClassifier, self).__init__(**kwargs)

        self.max_input_length = input_shape[0]
        self.lowest_level = lowest_level
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.rnn_type = rnn_type

    def init_model(self):
        input_layer = Input(shape=(self.max_input, ), )
        embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_size,
            input_length=self.max_input,
            embeddings_initializer=self.embedding,
            trainable=self.train_embedding
        )

        rnn_out, out_f, out_b = self.__get_rnn(embedding_layer)
        out_h = Concatenate([out_f, out_b])
        lower_attention = Attention("hierarchy")(out_h)
        lower_encoder = Model(input_layer, lower_attention)

        higher_input = Input(shape=(self.max_input_length, self.max_input), )
        dist_lower_encoder = TimeDistributed(lower_encoder)(higher_input)
        rnn_out, out_f, out_b = self.__get_rnn(dist_lower_encoder)
        out_h = Concatenate([out_f, out_b])
        higher_attention = Attention("hierarchy")(out_h)

        do = Dropout(self.dropout)(higher_attention)
        out = Dense(self.n_label, activation="softmax")(do)
        self.model = Model(higher_input, out)
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"]
        )
        self.model.summary()

    def __get_rnn(self, layer):
        if self.rnn_type == "lstm":
            rnn_out, out_f, _, out_b, _ = Bidirectional(
                LSTM(self.rnn_size, return_sequences=True, return_state=True)
            )(layer)
        elif self.rnn_type == "gru":
            rnn_out, out_f, out_b = Bidirectional(
                GRU(self.rnn_size, return_sequences=True, return_state=True)
            )(layer)
        return rnn_out, out_f, out_b

    def vectorized_input(self, tokenized_corpus):
        vector_input = np.zeros(
            (len(tokenized_corpus), self.max_input_length, self.max_input),
            dtype=np.int32
        )
        for i, doc in enumerate(tokenized_corpus):
            for j, sentence in enumerate(doc):
                for k, token in enumerate(sentence):
                    vector_input[i][j][k] = self.vocab.get(token, 0)
        return vector_input

    def get_class_param(self):
        return {
            "input_size": self.max_input,
            "l2i": self.label2idx,
            "i2l": self.idx2label,
            "vocab": self.vocab,
            "embedding_size": self.embedding_size,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "rnn_size": self.rnn_size,
            "dropout": self.dropout,
            "rnn_type": self.rnn_type,
            "max_input_length": self.max_input_length,
            "lowest_level": "char"
        }

    def load_class_param(self): pass
