from keras.layers import Layer, Embedding, Dense, Concatenate, Dot, Softmax
from keras.layers import Dropout, LayerNormalization
from keras import backend as K
import keras

import tensorflow as tf
from .relative_transformer_block import RelativeMultiAttention


class MultiAttentionOld(Layer):
    def __init__(self, n_heads, dim_k, dim_v, **kwargs):
        """
        Initialize multi head attention layer
        :param n_heads: int, number of attention heads
        :param dim_k: int, length of query & key vector
        :param dim_v: int, length of value vector
        """
        super(MultiAttentionOld, self).__init__(**kwargs)
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


class MultiAttention(Layer):
    def __init__(self, n_heads, dim_head, attention_do=0.5, **kwargs):
        """
        Initialize multi head attention layer
        :param n_heads: int, number of attention heads
        :param dim_head: int, length of query, key, and value for each head
        """
        super(MultiAttention, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.attention_do = attention_do

    def build(self, input_shape):
        self.W_q = Dense(self.n_heads*self.dim_head, use_bias=False)  # Wq
        self.W_k = Dense(self.n_heads*self.dim_head, use_bias=False)  # Wk
        self.W_v = Dense(self.n_heads*self.dim_head, use_bias=False)  # Wv

        # Get hidden/feature shape of the input
        self.W_o = Dense(input_shape[0][2], use_bias=False)

        self._dropout = Dropout(self.attention_do)

    def reshape_heads(self, vec):
        vec = K.reshape(
            vec, (-1, vec.shape[1], self.n_heads, self.dim_head)
        )
        return K.permute_dimensions(vec, (0, 2, 1, 3))

    def call(self, inputs, training):
        if len(inputs) == 2:
            query, key, value = inputs[0], inputs[1], inputs[1]
        else:
            query, key, value = inputs

        q_vec = self.W_q(query)  # B x S x (n_heads * d )
        k_vec = self.W_k(key)  # B x S x (n_heads * d )
        v_vec = self.W_v(value)  # B x S x (n_heads * d )
        # split heads and d
        q_vec = self.reshape_heads(q_vec)  # B x n_heads x S x d
        k_vec = self.reshape_heads(k_vec)  # B x n_heads x S x d
        v_vec = self.reshape_heads(v_vec)  # B x n_heads x S x d

        # Attention calculation
        # B x n_heads x S x S
        attention = tf.matmul(q_vec, k_vec, transpose_b=True)
        scale = tf.sqrt(float(self.dim_head))
        attention /= scale
        # TODO: Apply mask here
        attention = K.softmax(attention)
        attention = self._dropout(attention, training=training)

        # B x S x n_heads x d
        o_seq = tf.einsum("bhij,bhjd->bihd", attention, v_vec)
        # B x S x (n_heads * d)
        o_seq = K.reshape(o_seq, (
            -1, o_seq.shape[1], self.dim_head*self.n_heads
        ))
        output = self.W_o(o_seq)  # B x S x attention_d
        return output


class TransformerBlock(Layer):
    def __init__(
        self, dim_ff, n_heads, embed_dim, transformer_dropout=0.5,
        attention_dropout=0.5, attention_type="regular", scale=False, **kwargs
    ):
        """
        Initialize a transformer layer
        :param dim_ff: int, the size of hidden ffn unit
        :param n_heads: int, number of heads
        :param embed_dim: int, attention length (embedding length)
        :param transformer_dropout: float, dropout rate value in transformer
        :param attention_dropout: float, dropout rate value in attention
        :param attention_type: string, multihead attention type
            regular, regular multihead attention
            adaptive, TENER, A bit like transformer-XL albeit
                slightly different
        :param scale: boolean, attention scaling, only used at adaptive
            attention
        """
        super(TransformerBlock, self).__init__(**kwargs)
        if attention_type not in ["regular", "adaptive"]:
            raise ValueError("Invalid attention type")

        self.dim_ff = dim_ff
        self.n_heads = n_heads
        self.att_dim = embed_dim
        self.transformer_dropout = transformer_dropout
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.scale = scale

    def build(self, input_shape):
        if self.attention_type == "regular":
            self.mha = MultiAttention(
                self.n_heads, self.att_dim
            )
        elif self.attention_type == "adaptive":
            self.mha = RelativeMultiAttention(
                self.n_heads, self.att_dim, self.att_dropout, self.scale
            )
        self.ffn_hidden = Dense(self.dim_ff, activation="relu")
        self.ffn_out = Dense(self.att_dim)
        self.att_dropout = Dropout(self.transformer_dropout)
        self.att_layer_norm = LayerNormalization(epsilon=1e-6)
        self.fcn_dropout = Dropout(self.transformer_dropout)
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
        config["n_heads"] = self.n_heads
        config["embed_dim"] = self.att_dim
        config["transformer_dropout"] = self.transformer_dropout
        config["attention_dropout"] = self.attention_dropout
        config["attention_type"] = self.attention_type
        config["scale"] = self.scale
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
            trainable=self.trainable_embedding,
            mask_zero=True
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
