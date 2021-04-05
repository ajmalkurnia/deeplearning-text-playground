from keras.layers import Layer, Embedding, Dense, Concatenate, Dot, Softmax
from keras.layers import Dropout, LayerNormalization
from keras import backend as K
import keras

import tensorflow as tf


class MultiAttention(Layer):
    def __init__(self, n_heads, dim_k, dim_v, **kwargs):
        """
        Initialize multi head attention layer
        :param n_heads: int, number of attention heads
        :param dim_k: int, length of query & key vector
        :param dim_v: int, length of value vector
        """
        super(MultiAttention, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.dim_k = dim_k
        self.dim_v = dim_k

    def build(self, input_shape):
        self.attention_heads = [
            (
                Dense(self.dim_k, use_bias=False),  # Wq
                Dense(self.dim_k, use_bias=False),  # Wk
                Dense(self.dim_v, use_bias=False)   # Wv
            ) for i in range(self.n_heads)
        ]
        # Get hidden/feature shape of the input
        self.W_o = Dense(input_shape[0][2], use_bias=False)

    def attention_score(self, q, k, v):
        # scaled dot attention computation
        temp = Dot(axes=[2, 2])([q, k])
        scaled = tf.sqrt(float(q.shape[-1]))
        softmax = Softmax(axis=-1)(temp/scaled)
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
    def __init__(self, dim_ff, dropout, n_heads, embed_dim, **kwargs):
        """
        Initialize a transformer layer
        :param dim_ff: int, the size of hidden ffn unit
        :param dropout: float, dropout rate value
        :param n_heads: int, number of heads
        :param embed_dim: int, attention length (embedding length)
        """
        super(TransformerBlock, self).__init__(**kwargs)
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.n_heads = n_heads
        self.att_dim = embed_dim

    def build(self, input_shape):
        self.mha = MultiAttention(
            self.n_heads, self.att_dim, self.att_dim
        )
        self.ffn_hidden = Dense(self.dim_ff, activation="relu")
        self.ffn_out = Dense(self.att_dim)
        self.att_dropout = Dropout(self.dropout)
        self.att_layer_norm = LayerNormalization(epsilon=1e-6)
        self.fcn_dropout = Dropout(self.dropout)
        self.fcn_layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training, mask=None):
        if mask is not None:
            casted_mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
            inputs *= casted_mask
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
        self, maxlen, vocab_size, embed_dim=256,
        token_embed_matrix="glorot_uniform", pos_embedding_init=True,
        trainable_embedding=True, **kwargs
    ):
        """
        Initialize transfomer token+positional embedding
        :param maxlen: int, maximum sequence length
        :param vocab_size: int, the size of token emedding vocabulary
        :param embed_dim: int, embedding dimension
        :param token_embed_matrix: numpy array/string,
            token embedding initializer,
            if numpy array is given the embed_dim must be consistent
                with matrix shape
            if string is given use valid keras initializer
        :param pos_embedding_init: bool, initialize positional embedding
            if true use the sincos function,
            if false use the glorot_uniform
        """
        super(TransformerEmbedding, self).__init__(**kwargs)
        self.max_len = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_initializer = token_embed_matrix

        if pos_embedding_init:
            self.pos_initializer = keras.initializers.Constant(
                self.init_pos_emb((maxlen, embed_dim), float)
            )
        else:
            self.pos_initializer = "glorot_uniform"
        self.trainable_embedding = trainable_embedding

    def build(self, init_shape):
        self.token_emb = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_dim,
            embeddings_initializer=self.token_initializer,
            trainable=self.trainable_embedding
            # mask_zero=True
        )

        self.pos_emb = Embedding(
            input_dim=self.max_len,
            output_dim=self.embed_dim,
            embeddings_initializer=self.pos_initializer,
            trainable=self.trainable_embedding
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
