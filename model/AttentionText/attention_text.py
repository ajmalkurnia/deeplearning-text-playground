from keras.layers import Layer, Activation, Dot, Dense
from keras.layers import RepeatVector, Lambda
from keras import backend as K
import tensorflow as tf
import numpy as np


class Attention(Layer):
    def __init__(
        self, score="", return_attention=False, penalty=1.0,
        feature=0, max_length=0, **kwargs
    ):
        """
        Construct Attention Layer
        :param score: string, attention scoring method
            available options {dot, scaled, general, location, add, self}
            dot, general, location attention is based on Luong, 2015
            scaled attention is based on Vaswani, 2017, Transformer's attention
            add attention is based on Bahdanau, 2014, also known as concat
            self attention is based on Zhouhan, 2016
        :param return attention: boolean,
            If true it will send the attention score
            (usually for visualization purpose)
        :param penalty: float, additional loss factor
            (only used on self attention)
        :param feature: int, hidden/feature dimension, for loading model
        :param max_length: int, maximum sequence length, for loading model
        """
        self.ATTENTION_SCORE_MAP = {
            "dot": self.dot_score,
            "scaled": self.scaled_score,
            "general": self.general_score,
            "location": self.location_score,
            "add": self.additive_score,
            "self": self.self_score,
            "hierarchy": self.hierarchy_score
        }
        if score not in self.ATTENTION_SCORE_MAP:
            raise ValueError(
                "Invalid score parameter,\
                valid scoring are dot|scaled|general|location|add|self"
            )
        self.score = score
        self.return_attention = return_attention
        self.score_function = self.ATTENTION_SCORE_MAP[score]
        self.penalty = penalty

        # will be initialized upon build
        self.feature = feature
        self.maxlen = max_length
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        try:
            self.feature = input_shape[0][-1]
            self.maxlen = input_shape[0][1]
        except TypeError:
            self.feature = input_shape[-1]
            self.maxlen = input_shape[1]

        if self.score in ["general", "location"]:
            self.W1 = Dense(self.feature, name="key_att", use_bias=False)
        elif self.score == "add":
            self.W1 = Dense(self.feature, name="key_att", use_bias=False)
            self.W2 = Dense(self.feature, name="query_att", use_bias=False)
            self.V = Dense(1)
        elif self.score == "self":
            self.W1 = Dense(self.feature, activation="tanh", use_bias=False)
            self.W2 = Dense(self.feature, name="sf_W", use_bias=False)
        elif self.score == "hierarchy":
            self.W1 = Dense((self.feature), activation="tanh", use_bias=False)

        super(Attention, self).build(input_shape)

    def dot_score(self, query, key):
        return Dot(axes=[2, 2])(query, key)

    def scaled_score(self, query, key):
        scaler = np.sqrt(float(query.shape[2]))
        return self.dot_score(query, key)/scaler

    def general_score(self, query, key):
        return Dot(axes=[2, 2])([query, self.W1(key)])

    def location_score(self, query, key):
        score = self.W1(key)
        score = Activation("softmax")(score)
        score = RepeatVector(query.shape[1])(score)
        score = tf.reduce_sum(score, axis=-1)
        return tf.expand_dims(score, axis=-1)

    def additive_score(self, query, key):
        return self.V(tf.nn.tanh(self.W2(query) + self.W1(key)))

    def self_score(self, query, key):
        return self.W2(self.W1(query))

    def hierarchy_score(self, query, key):
        return self.W1(query)

    def _compute_additional_loss(self, attention_weights):
        """
        Compute Additional Loss on "self" attention = ||A.At-I||
        :param attention_weight: normalized attention score
        """
        # A . At
        product = Dot(axes=[1, 1])([attention_weights, attention_weights])
        # I
        identity = tf.eye(self.feature)
        # ||A.At-I||
        frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(product - identity)))
        # actual loss = penalty_factor * ||A.At-I||
        self.add_loss(self.penalty * frobenius_norm)

    def call(self, inputs, mask=None):
        """
        Run attention mechanism
        :param inputs: a list [query, key]/tensor query,
            Query = Source = Output of each encoder state,
                (Batch, Sequence, Feature/Hidden)
            Key = Target = Decoder state/last decoder state, not used on self
                (Batch, Feature/Hidden)
        :return [context_vector, attention_weight] if return_attention is True
        :return context_vector if return_attention is set False
            on 'self' attention the output shape is:
                context_vector, (Batch, Feature*Sequence)
                attention_weight, (Batch, Sequence, Feature)
            on other score the output shape is :
                context_vector, shape (Batch, Feature)
                attention_weight, shape (Batch, Sequence, 1)
        """
        if isinstance(inputs, list):
            query, key = inputs
            mask = mask[0]
        else:
            query, key = inputs, None

        if key is not None:
            key = tf.expand_dims(key, 1)

        if mask is not None:
            casted_mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
            query *= casted_mask
            # 'self' scoring will return (batch, seq, hid)
            # others will return (bacth, seq, 1)
            score = self.score_function(query, key)
            score *= casted_mask
        else:
            score = self.score_function(query, key)
        # normalized attention score
        attention_weights = Activation("softmax")(score)

        # attention matrix
        attention_matrix = Dot(axes=[1, 1])([attention_weights, query])

        if self.score == "self":
            # Add loss for penalization
            self._compute_additional_loss((attention_weights))
        context_vector = Lambda(
                lambda x: K.sum(x, axis=1)
            )(attention_matrix)

        if self.return_attention:
            return [context_vector, attention_weights]
        else:
            return context_vector

    def compute_output_shape(self, input_shape):
        if self.return_attention and self.score == "self":
            return [
                (None, self.feature),
                (None, self.maxlen, self.feature)
            ]
        elif self.return_attention:
            return [(None, self.feature), (None, self.maxlen, 1)]
        else:
            return (None, self.feature)

    def get_config(self):
        config = super(Attention, self).get_config()
        config["return_attention"] = self.return_attention
        config["score"] = self.score
        config["penalty"] = self.penalty
        return config


class CharTagAttention(Layer):
    def __init__(self, feature, max_length, **kwargs):
        super(CharTagAttention, self).__init__(**kwargs)
        self.feature = feature
        self.maxlen = max_length

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.feature))

    def call(self, query, mask):
        att_score = K(axes=[2, 1])([query, self.W])
        att_score = Activation("softmax")(att_score)
        att_mat = Dot(axes=[2, 1])([K.transpose(query), att_score])
        vector = Lambda(lambda x: K.sum(x, axis=1))(att_mat)
        return vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    # def get_config(self): pass
