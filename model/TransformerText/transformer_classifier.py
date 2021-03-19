from keras.layers import Layer, Embedding, Dense, Concatenate, Dot, Activation
from keras.layers import Dropout, LayerNormalization
from keras.layers import Input, GlobalAveragePooling1D
from keras.models import Model
from keras import backend as K
from model.base_classifier import BaseClassifier
import keras

import tensorflow as tf


class MultiHeadAttention(Layer):
    # TODO: Test MultiHeads Attention
    def __init__(self, n_heads, dim_k, dim_v, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.dim_k = dim_k
        self.dim_v = dim_k

    def build(self, input_shape):
        self.attention_heads = [
            (
                Dense(self.dim_k),  # Wq
                Dense(self.dim_k),  # Wk
                Dense(self.dim_v)   # Wv
            ) for i in range(self.n_heads)
        ]
        # Get hidden/feature shape of the input
        self.W_o = Dense(input_shape[0][2])

    def attention_score(self, q, k, v):
        temp = Dot(axes=[2, 2])([q, k])
        scaled = tf.sqrt(float(q.shape[-1]))
        softmax = Activation("softmax")(temp/scaled)
        return Dot(axes=[2, 1])([softmax, v])

    def call(self, inputs):
        if len(inputs) == 2:
            query, key, value = inputs[0], inputs[1], inputs[1]
        else:
            query, key, value = inputs

        attention = Concatenate()(
            [
                self.attention_score(Wq(query), Wk(key), Wv(value))
                for Wq, Wk, Wv in self.attention_heads
            ]
        )
        output = self.W_o(attention)
        return output


class TransformerBlock(Layer):
    # TODO: Test Encoder Block
    def __init__(self, dim_ff, dropout, n_heads, embed_dim, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.n_heads = n_heads
        self.att_dim = embed_dim

    def build(self, input_shape):
        self.mha = MultiHeadAttention(self.n_heads, self.att_dim, self.att_dim)
        self.ffn_hidden = Dense(self.dim_ff, activation="relu")
        self.ffn_out = Dense(self.att_dim)
        self.att_dropout = Dropout(self.dropout)
        self.att_layer_norm = LayerNormalization(epsilon=1e-6)
        self.fcn_dropout = Dropout(self.dropout)
        self.fcn_layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training):
        mha_out = self.mha([inputs, inputs])
        mha_out = self.att_dropout(mha_out, training=training)
        mha_out = self.att_layer_norm(inputs + mha_out)

        fcn_out = self.ffn_out(self.ffn_hidden(mha_out))
        fcn_out = self.fcn_dropout(fcn_out, training=training)

        return self.fcn_layer_norm(mha_out + fcn_out)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config["dim_ff"] = self.dim_ff
        config["dropout"] = self.dropout
        config["n_heads"] = self.n_heads
        config["embed_dim"] = self.att_dim
        return config


class TransformerEmbedding(Layer):
    def __init__(
        self, maxlen, vocab_size, embed_dim=256, token_embed_matrix=None,
        pos_embedding_init=True, **kwargs
    ):
        super(TransformerEmbedding, self).__init__(**kwargs)
        self.max_len = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        if token_embed_matrix is None:
            token_initializer = 'glorot_uniform'
        else:
            token_initializer = keras.initializers.Constant(token_embed_matrix)

        if pos_embedding_init:
            pos_initializer = keras.initializers.Constant(
                self.init_pos_emb((maxlen, embed_dim), float)
            )
        else:
            pos_initializer = "glorot_uniform"

        self.token_emb = Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            embeddings_initializer=token_initializer
        )

        self.pos_emb = Embedding(
            input_dim=maxlen,
            output_dim=embed_dim,
            embeddings_initializer=pos_initializer
        )

    def init_pos_emb(self, shape, dtype):
        in_dim, out_dim = shape
        pos = tf.reshape(tf.range(0, in_dim, dtype=dtype), [in_dim, 1])
        dim = tf.reshape(tf.range(0, out_dim, dtype=dtype), [1, out_dim])
        phase = pos / 1e4 ** (dim // out_dim)
        return tf.where(dim % 2 == 0, K.sin(phase), K.cos(phase))

    def call(self, x):
        # B, S
        maxlen = tf.shape(x)[-1]
        # change This into sin cos equation
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        # B, S, H
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super(TransformerEmbedding, self).get_config()
        config["maxlen"] = self.max_len
        config["vocab_size"] = self.vocab_size
        config["embed_dim"] = self.embed_dim
        return config


class TransformerClassifier(BaseClassifier):
    # TODO: Test the classifier
    def __init__(
        self, n_blocks=1, dim_ff=128, dropout=0.3, n_heads=6,
        attention_dim=256, pos_embedding_init=True,
        fcn_layers=[(128, 0.1, "relu")], **kwargs
    ):
        super(TransformerClassifier, self).__init__(**kwargs)
        self.n_blocks = n_blocks
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.n_heads = n_heads
        self.attention_dim = attention_dim
        self.pos_embedding_init = pos_embedding_init
        self.fcn_layers = fcn_layers

    def init_model(self):
        input_layer = Input(shape=(self.max_input, ))

        if self.embedding.shape[1] == self.vocab_size:
            self.embedding = None
        if self.embedding is not None:
            self.attention_dim = self.embedding.shape[1]

        self.model = TransformerEmbedding(
            self.max_input, self.vocab_size, self.attention_dim,
            self.embedding, self.pos_embedding_init
        )(input_layer)
        for _ in range(self.n_blocks):
            self.model = TransformerBlock(
                self.dim_ff, self.dropout, self.n_heads, self.attention_dim
            )(self.model)
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

    def init_embedding(self):
        if self.embedding_file:
            super().init_wv_embedding()

    def get_class_param(self):
        return {
            "input_size": self.max_input,
            "l2i": self.label2idx,
            "i2l": self.idx2label,
            "vocab": self.vocab,
            "embedding_size": self.embedding_size,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "dropout": self.dropout,
            "n_blocks": self.n_blocks,
            "dim_ff": self.dim_ff,
            "n_heads": self.n_heads,
            "attention_dim": self.attention_dim,
            "fcn_layers": self.fcn_layers,
            "pos_embedding": self.pos_embedding_init
        }

    def load_class_param(self, class_param):
        self.max_input = class_param["input_size"]
        self.label2idx = class_param["l2i"]
        self.idx2label = class_param["i2l"]
        self.vocab = class_param["vocab"]
        self.embedding_size = class_param["embedding_size"]
        self.optimizer = class_param["optimizer"]
        self.loss = class_param["loss"]
        self.dropout = class_param["dropout"]
        self.n_blocks = class_param["n_blocks"]
        self.dim_ff = class_param["dim_ff"]
        self.n_heads = class_param["n_heads"]
        self.attention_dim = class_param["attention_dim"]
        self.fcn_layers = class_param["fcn_layers"]
        self.pos_embedding_init = class_param["pos_embedding"]
