from keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional, GRU
from keras.layers import Concatenate
from keras.models import Model, Input

from model.AttentionText.attention_text import Attention
from model.base_classifier import BaseClassifier


class RNNClassifier(BaseClassifier):
    def __init__(
        self, rnn_size=100, dropout=0.5, rnn_type="lstm", attention=None,
        fcn_layers=[(128, 0.1, "relu")], **kwargs
    ):
        """
            :param rnn_size: int, RNN hidden unit
            :param dropout: float, [0.0, 1.0] dropout just before softmax layer
            :param rnn_type: string, RNN memory type, "gru"/"lstm"
            :param attention: string, attention type choice available:
                dot|scale|general|location|add|self,
                set None do not want to use attention mechanism
        """
        # self.__doc__ = BaseClassifier.__doc__
        super(RNNClassifier, self).__init__(**kwargs)
        if rnn_type not in ["lstm", "gru"]:
            raise ValueError("Invalid RNN type")
        valid_attention = ["dot", "scaled",
                           "general", "location", "add", "self", None]
        if attention not in valid_attention:
            raise ValueError("Invalid attention mechanism")

        self.rnn_size = rnn_size
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.attention = attention
        self.fcn_layers = fcn_layers
        if self.attention:
            self.return_seq = True
        else:
            self.return_seq = False

    def init_model(self):
        """
        Initialized RNN Model
        """
        # Input
        input_layer = Input(shape=(self.max_input,))
        # Embedding_layer
        embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_size,
            input_length=self.max_input,
            embeddings_initializer=self.embedding,
            trainable=self.train_embedding,
            mask_zero=True
        )
        self.model = embedding_layer(input_layer)

        # RNN
        if self.rnn_type == "lstm":
            rnn_output = Bidirectional(LSTM(
                self.rnn_size, return_sequences=self.return_seq,
                return_state=self.return_seq
            ))(self.model)
            if self.return_seq:
                self.model, forward_h, _, backward_h, _ = rnn_output
            else:
                self.model = rnn_output
        else:
            rnn_output = Bidirectional(GRU(
                self.rnn_size, return_sequences=self.return_seq,
                return_state=self.return_seq
            ))(self.model)
            if self.return_seq:
                self.model, forward_h, backward_h, = rnn_output
            else:
                self.model = rnn_output

        # Attention mechanism
        if self.attention:
            state_h = Concatenate()([forward_h, backward_h])
            attention_layer = Attention(self.attention, True)
            self.model, x = attention_layer([self.model, state_h])

        # Dropout
        self.model = Dropout(self.dropout)(self.model)
        for units, do_rate, activation in self.fcn_layers:
            self.model = Dense(units, activation=activation)(self.model)
            self.model = Dropout(do_rate)(self.model)
        # Last Layer
        out = Dense(self.n_label, activation="softmax")(self.model)
        self.model = Model(input_layer, out)
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"]
        )
        self.model.summary()

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
            "attention": self.attention,
            "return_seq": self.return_seq
        }

    @staticmethod
    def get_construtor_param(param):
        return {
            "input_size": param["input_size"],
            "vocab": param["vocab"],
            "embedding_size": param["embedding_size"],
            "rnn_size": param["rnn_size"],
            "dropout": param["dropout"],
            "rnn_type": param["rnn_type"],
            "attention": param["attention"]
        }
