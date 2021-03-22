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
        kwargs["input_size"] = input_shape[1]
        super(HANClassifier, self).__init__(**kwargs)

        self.max_input_length = input_shape[0]
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.rnn_type = rnn_type

    def init_model(self):
        input_layer = Input(shape=(self.max_input, ))
        embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_size,
            input_length=self.max_input,
            embeddings_initializer=self.embedding,
            trainable=self.train_embedding
        )(input_layer)

        rnn_out = self.__get_rnn(embedding_layer)
        # out_h = Concatenate()([out_f, out_b])
        lower_attention = Attention("hierarchy")([rnn_out])
        lower_encoder = Model(inputs=input_layer, outputs=lower_attention)
        lower_encoder.summary()

        higher_input = Input(shape=(self.max_input_length, self.max_input))
        dist_lower_encoder = TimeDistributed(lower_encoder)(higher_input)
        rnn_out = self.__get_rnn(dist_lower_encoder)
        # out_h = Concatenate()])
        higher_attention = Attention("hierarchy")([rnn_out])

        do = Dropout(self.dropout)(higher_attention)
        out = Dense(self.n_label, activation="softmax")(do)
        self.model = Model(higher_input, out)
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"],
            sample_weight_mode="temporal"
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
            "optimizer": self.optimizer,
            "loss": self.loss,
            "rnn_size": self.rnn_size,
            "dropout": self.dropout,
            "rnn_type": self.rnn_type,
            "max_input_length": self.max_input_length
        }

    def load_class_param(self, class_param):
        self.max_input = class_param["input_size"]
        self.label2idx = class_param["l2i"]
        self.idx2label = class_param["i2l"]
        self.vocab = class_param["vocab"]
        self.embedding_size = class_param["embedding_size"]
        self.optimizer = class_param["optimizer"]
        self.loss = class_param["loss"]
        self.rnn_size = class_param["rnn_size"]
        self.dropout = class_param["dropout"]
        self.rnn_type = class_param["rnn_type"]
        self.max_input_length = class_param["max_input_length"]
