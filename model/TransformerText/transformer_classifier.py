from keras.layers import Dense, Dropout, Lambda
from keras.layers import Input, GlobalAveragePooling1D  # , MultiHeadAttention
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from model.base_classifier import BaseClassifier
from model.TransformerText.transformer_block import (
    TransformerEmbedding, TransformerBlock
)


class TransformerClassifier(BaseClassifier):
    def __init__(
        self, n_blocks=1, dim_ff=128, dropout=0.3, n_heads=6,
        attention_dim=256, pos_embedding_init=True,
        fcn_layers=[(128, 0.1, "relu")],
        sequence_embedding="global_avg", **kwargs
    ):
        """
        Transformer classifier's construction method
            :param n_blocks: int, number of transformer stack
            :param dim_ff: int, hidden unit on fcn layer in transformer
            :param dropout: float, dropout value
            :param n_heads: int, number of attention heads
            :param attention_dim: int, number of attention dimension
                this value will be overidden if using custom embedding matrix
            :param pos_embedding_init: bool, Initialize posiitonal embedding
                with sincos function, or else will be initialize with
                "glorot_uniform"
            :param fcn_layers: list of tupple, configuration of each
                fcn layer after transformer, each tupple consist of:
                    [int] number of units,
                    [float] dropout after fcn layer,
                    [string] activation function
            :param sequence_embedding: string, a method how to get
                representation of entire sequences extract, available option:
                cls, prepend [CLS] token in the sequence, then take
                    attention embedding of [CLS] as sequence embedding
                    (BERT style)
                global_avg, use GlobalAveragePool1D
        """
        # self.__doc__ = BaseClassifier.__doc__
        super(TransformerClassifier, self).__init__(**kwargs)
        if sequence_embedding not in ["global_avg", "cls"]:
            raise ValueError("Invalid sequence embedding")

        self.n_blocks = n_blocks
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.n_heads = n_heads
        self.attention_dim = attention_dim
        self.pos_embedding_init = pos_embedding_init
        self.fcn_layers = fcn_layers
        self.sequence_embedding = sequence_embedding

    def init_model(self):
        input_layer = Input(shape=(self.max_input, ))
        if not isinstance(self.embedding, str):
            self.attention_dim = self.embedding.value.shape[1]

        self.model = TransformerEmbedding(
            self.max_input, self.vocab_size, self.attention_dim,
            self.embedding, self.pos_embedding_init, self.train_embedding
        )(input_layer)
        for _ in range(self.n_blocks):
            self.model = TransformerBlock(
                self.dim_ff, self.dropout, self.n_heads, self.attention_dim
            )(self.model)

        if self.sequence_embedding == "cls":
            self.model = Lambda(lambda x: x[:, 0, :])(self.model)
        else:
            self.model = GlobalAveragePooling1D()(self.model)

        self.model = Dropout(self.dropout)(self.model)
        for units, do_rate, activation in self.fcn_layers:
            self.model = Dense(units, activation=activation)(self.model)
            self.model = Dropout(do_rate)(self.model)
        output = Dense(self.n_label, "softmax")(self.model)
        self.model = Model(input_layer, output)
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"]
        )
        self.model.summary()

    def vectorized_input(self, corpus):
        v_input = []
        for text in corpus:
            idx_list = []
            if self.sequence_embedding == "cls":
                text = ["[CLS]"] + text
            for idx, token in enumerate(text):
                if token in self.vocab:
                    idx_list.append(self.vocab[token])
                else:
                    idx_list.append(0)
            v_input.append(idx_list)
        return pad_sequences(v_input, self.max_input, padding='post')

    def add_special_token(self):
        if self.sequence_embedding == "cls":
            return ["[UNK]", "[CLS]"]
        else:
            return ["[UNK]"]

    def get_class_param(self):
        return {
            "input_size": self.max_input,
            "l2i": self.label2idx,
            "i2l": self.idx2label,
            "vocab": self.vocab,
            "embedding_size": self.embedding_size,
            "dropout": self.dropout,
            "n_blocks": self.n_blocks,
            "dim_ff": self.dim_ff,
            "n_heads": self.n_heads,
            "attention_dim": self.attention_dim,
            "fcn_layers": self.fcn_layers,
            "pos_embedding": self.pos_embedding_init,
            "sequence_embedding": self.sequence_embedding
        }

    @staticmethod
    def get_construtor_param(param):
        return {
            "input_size": param["input_size"],
            "vocab": param["vocab"],
            "embedding_size": param["embedding_size"],
            "n_blocks": param["n_blocks"],
            "dim_ff": param["dim_ff"],
            "dropout": param["dropout"],
            "n_heads": param["n_heads"],
            "attention_dim": param["attention_dim"],
            "pos_embedding_init": param["pos_embedding"],
            "fcn_layers": param["fcn_layers"],
            "sequence_embedding": param["sequence_embedding"]
        }
