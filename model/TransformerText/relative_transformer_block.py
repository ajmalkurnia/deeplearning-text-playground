from keras.layers import Layer, Dropout
from keras.initializers import Constant
from keras import backend as K
# import keras
import math
import tensorflow as tf


class RelativeMultiAttention(Layer):
    def __init__(
        self, n_heads, dim_head, attention_do=0.5, scale=False, **kwargs
    ):
        """
        Initialize Relative multi head attention layer
        Based on:
        https://arxiv.org/abs/1911.04474

        :param n_heads: int, number of attention heads
        :param dim_head: int, length of query, key, and value for each head
        :param attention_do: float, dropout rate for attention
        """
        super(RelativeMultiAttention, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.attention_do = attention_do
        if scale:
            self.scale = tf.sqrt(float(self.dim_head))
        else:
            self.scale = 1
        # Init relative embedding

    def build(self, input_shape):
        self.W_q = self.add_weight(
            shape=(input_shape[0][2], self.n_heads*self.dim_head), name="wq"
        )  # Wq
        self.W_k = self.add_weight(
            shape=(input_shape[0][2], self.n_heads*self.dim_head), name="wk"
        )  # Wk
        self.W_v = self.add_weight(
            shape=(input_shape[0][2], self.n_heads*self.dim_head), name="wv"
        )  # Wv

        # Get hidden/feature shape of the input
        self.W_o = self.add_weight(
            shape=(self.n_heads*self.dim_head, input_shape[0][2]),
            name="o"
        )

        # BUILD
        self.u = self.add_weight(
            shape=(self.n_heads, self.dim_head),
            initializer="glorot_normal", trainable=True, name="u"
        )
        self.v = self.add_weight(
            shape=(self.n_heads, self.dim_head),
            initializer="glorot_normal", trainable=True, name="v"
        )

        self.pos_embed = self.add_weight(
            shape=(2*input_shape[0][1], self.dim_head),
            initializer=self.init_pos_emb(
                (input_shape[0][1], self.dim_head), dtype=tf.float32
            ),
            trainable=False, name="pos"
        )

        self._dropout = Dropout(self.attention_do)

    def init_pos_emb(self, shape, dtype):
        maxlen, out_dim = shape
        pos = tf.range(-maxlen, maxlen, dtype=dtype)
        dim = tf.range(0, out_dim//2, dtype=dtype)
        inv = math.log(10000)/((out_dim//2)-1)
        inv = tf.exp(dim * -inv)
        phase = tf.einsum("i,j->ij", pos, inv)
        return Constant(tf.concat([tf.sin(phase), tf.cos(phase)], -1))

    def reshape_heads(self, vec):
        vec = K.reshape(
            vec, (-1, vec.shape[1], self.n_heads, self.dim_head)
        )
        return K.permute_dimensions(vec, (0, 2, 1, 3))

    def shift(self, x):
        x_s = tf.shape(x)
        pad = [[0, 0], [0, 0], [0, 0], [1, 1]]
        x = tf.pad(x, pad)[:, :, :, 1:]
        x = tf.reshape(x, [x_s[0], x_s[1], -1, x_s[2]])
        x = tf.reshape(x[:, :, :-1], [x_s[0], x_s[1], x_s[2], -1])
        return x[:, :, :, x_s[2]:]

    def t_shift(self, x):
        x_s = tf.shape(x)
        pad = [[0, 0], [0, 0], [0, 0], [1, 1]]
        x = tf.pad(x, pad)[:, :, :, 1:]
        x = tf.reshape(x, [x_s[0], x_s[1], -1, x_s[2]])
        x = tf.gather(x, tf.range(x_s[2])*2+1, axis=2)
        return tf.transpose(x, [0, 1, 3, 2])

    def call(self, inputs, training):
        if len(inputs) == 2:
            query, key, value = inputs[0], inputs[1], inputs[1]
        else:
            query, key, value = inputs

        q_vec = K.dot(value, self.W_q)  # B x S x (n_heads * d )
        k_vec = K.dot(value, self.W_k)  # B x S x (n_heads * d )
        v_vec = K.dot(value, self.W_v)  # B x S x (n_heads * d )
        # split heads and d
        q_vec = self.reshape_heads(q_vec)  # B x n_heads x S x d
        k_vec = self.reshape_heads(k_vec)  # B x n_heads x S x d
        v_vec = self.reshape_heads(v_vec)  # B x n_heads x S x d

        # Attention calculation
        # B x n_heads x S x S
        #  a  +  b  + c   + d + e (?)
        # q*k + q*r + u*k + v*r
        # q*k + u*k + q*r + v*r
        # (q+u) * k + q*r + v*r
        term_ac = q_vec + self.u[:, None]
        term_ac = tf.einsum(
            "bnqd,bnkd->bnqk", term_ac, k_vec, optimize="optimal"
        )
        term_b = tf.einsum(
            "nd,ld->nl", self.v, self.pos_embed, optimize="optimal"
        )[None, :, None]
        term_d = tf.einsum(
            "bnqd,ld->bnql", q_vec, self.pos_embed, optimize="optimal"
        )
        term_e = tf.einsum(
            "bnqd,ld->bnql", k_vec, self.pos_embed, optimize="optimal"
        )
        term_bde = self.shift(term_b + term_d) + self.t_shift(term_e)
        attention = term_ac + term_bde

        attention /= self.scale
        # TODO: Apply mask here
        attention = K.softmax(attention)
        attention = self._dropout(attention, training=training)

        # B x S x n_heads x d
        o_seq = tf.einsum("bhij,bhjd->bihd", attention, v_vec)
        # B x S x (n_heads * d)
        o_seq = K.reshape(o_seq, (
            -1, o_seq.shape[1], self.dim_head*self.n_heads
        ))
        # B x S x attention_d
        output = tf.einsum("bhd,dl->bhl", o_seq, self.W_o)
        return output
