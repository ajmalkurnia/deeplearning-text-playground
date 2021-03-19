from keras.layers import Layer, Embedding, Dense, Concatenate, Dot, Activation
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
                Dense(self.dim_k, name="W_query"),
                Dense(self.dim_k, name="W_key"),
                Dense(self.dim_v, name="W_value")
            ) for _ in range(self.n_heads)
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


class TransformerEncoderBlock(Layer):
    # TODO: Implement Encoder Block
    def __init__(): pass
    def build(): pass
    def call(): pass
    # def compute_output_shape(): pass


class TransformersEmbedding(Layer):
    def __init__(
        self, maxlen, vocab_size, embed_dim, token_embed_matrix=None,
        pos_embedding_init=True
    ):
        super(TransformersEmbedding, self).__init__()
        if token_embed_matrix is None:
            token_initializer = 'glorot_uniform'
        else:
            token_initializer = keras.initializers.Constant(token_embed_matrix)

        if pos_embedding_init:
            pos_initializer = keras.initializers.Constant(self.init_pos_emb)
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


class TransformerClassifier(BaseClassifier):
    # TODO: Implement the classifier
    def __init__(): pass
    def init_model(): pass
    def get_class_param(): pass
    def load_class_param(): pass
