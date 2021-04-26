from keras.layers import Embedding, Input, Bidirectional, LSTM, Dense
from keras.layers import Layer, Concatenate, Add, Dropout
from keras.metrics import Mean
from keras.models import Model
from keras.losses import CategoricalCrossentropy

from keras import backend as K
import tensorflow as tf
import numpy as np
import string
from model.base_tagger import BaseTagger


class CharToWord(Layer):
    def __init__(self, seq_length, word_border, **kwargs):
        super(CharToWord, self).__init__(**kwargs)
        self.seq_length = seq_length
        self.word_border = word_border

    def call(self, inp, mask):
        rnn_out, char_seq = inp
        out_shape = K.shape(rnn_out)
        word_border = char_seq == self.word_border
        # Shift
        pad = tf.cast(tf.zeros((K.shape(char_seq)[0], 1)), dtype=bool)
        bos = tf.concat([pad, word_border[:, :-1]], axis=1)
        bos = tf.ragged.boolean_mask(
            rnn_out, bos
        ).to_tensor(0., shape=(out_shape[0], self.seq_length, out_shape[2]))

        eos = tf.concat([word_border[:, 1:], pad], axis=1)
        eos = tf.ragged.boolean_mask(
            rnn_out, eos
        ).to_tensor(0., shape=(out_shape[0], self.seq_length, out_shape[2]))
        # This part does not follow the original paper
        # The paper uses both hidden state and carry state
        # In keras, carry state cannot be retrieved mid sequence
        #   without modifying the RNN
        # So, this implementation only uses hidden state
        word_embed = Concatenate(axis=-1)([bos, eos])
        return word_embed

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.seq_length, input_shape[0][2])

    def get_config(self):
        config = super(CharToWord, self).get_config()
        config["word_border"] = self.word_border
        return config


class MetaModelWrapper(Model):
    def __init__(
        self, char_model, word_model, meta_model, target_label, **kwargs
    ):
        super(MetaModelWrapper, self).__init__(**kwargs)
        # tf.enable_eager_execution()
        self.char_model = char_model
        self.word_model = word_model
        self.meta_model = meta_model
        self.char_out = Dense(target_label)
        self.word_out = Dense(target_label)

    def call(self, input_data):
        X = input_data
        w_embed = self.word_model(X["word"])
        c_embed = self.char_model(X["char"])
        m_input = tf.concat([c_embed, w_embed], axis=-1)
        m_pred = self.meta_model(m_input)
        return m_pred

    def compile(
        self, char_optimizer, word_optimizer, meta_optimizer, loss,
        **kwargs  # , metric
    ):
        super(MetaModelWrapper, self).compile(**kwargs)
        self.c_opt = char_optimizer
        self.w_opt = word_optimizer
        self.m_opt = meta_optimizer
        self.loss_fn = loss
        # self.mertic = metric
        self.c_loss = Mean("c_loss")
        self.w_loss = Mean("w_loss")
        self.m_loss = Mean("loss")

    @property
    def metrics(self):
        return [self.c_loss, self.w_loss, self.m_loss]

    def train_step(self, input_data):
        X, y = input_data

        with tf.GradientTape() as wtape:
            w_embed = self.word_model(X["word"])
            w_pred = self.word_out(w_embed)
            # w_embed = self.word_model.layers[-2].output
            w_loss = self.loss_fn(y, w_pred)
        wgrads = wtape.gradient(w_loss, self.word_model.trainable_weights)
        self.w_opt.apply_gradients(
            zip(wgrads, self.word_model.trainable_weights)
        )

        with tf.GradientTape() as ctape:
            c_embed = self.char_model(X["char"])
            c_pred = self.char_out(c_embed)
            # c_pred = weights here
            c_loss = self.loss_fn(y, c_pred)
        cgrads = ctape.gradient(c_loss, self.char_model.trainable_weights)
        self.c_opt.apply_gradients(
            zip(cgrads, self.char_model.trainable_weights)
        )

        m_input = tf.concat([c_embed, w_embed], axis=-1)
        with tf.GradientTape() as tape:
            m_pred = self.meta_model(m_input)
            m_loss = self.loss_fn(y, m_pred)
        grads = tape.gradient(
            m_loss, self.meta_model.trainable_variables
        )
        self.m_opt.apply_gradients(
            zip(grads, self.meta_model.trainable_variables)
        )

        self.c_loss.update_state(c_loss)
        self.w_loss.update_state(w_loss)
        self.m_loss.update_state(m_loss)
        return {
            "c_loss": self.c_loss.result(),
            "w_loss": self.w_loss.result(),
            "loss": self.m_loss.result()
        }

    def test_step(self, input_data):
        X, y = input_data

        w_embed = self.word_model(X["word"])
        w_pred = self.word_out(w_embed)
        w_loss = self.loss_fn(y, w_pred)
        c_embed = self.char_model(X["char"])
        c_pred = self.char_out(c_embed)
        c_loss = self.loss_fn(y, c_pred)
        m_input = tf.concat([c_embed, w_embed], axis=-1)
        m_pred = self.meta_model(m_input)
        m_loss = self.loss_fn(y, m_pred)

        self.c_loss.update_state(c_loss)
        self.w_loss.update_state(w_loss)
        self.m_loss.update_state(m_loss)

        return {
            "c_loss": self.c_loss.result(),
            "w_loss": self.w_loss.result(),
            "loss": self.m_loss.result()
        }


class MetaBiLSTMTagger(BaseTagger):
    def __init__(
        self, char_seq_length=550, char_embedding_size=100, char_rnn_units=100,
        char_rd=0.5, char_ed=0.5, char_dense=256, word_rnn_units=100,
        word_rd=0.5, word_dense=256, word_ed=0.5,
        meta_rnn_units=100, meta_rd=0.5, meta_ed=0.5, meta_dense=256,
        **kwargs
    ):
        super(MetaBiLSTMTagger, self).__init__(**kwargs)
        self.char_seq_length = char_seq_length
        self.char_embedding_size = char_embedding_size
        self.char_rnn_units = char_rnn_units
        self.char_rd = char_rd
        self.char_ed = char_ed
        self.char_dense = char_dense
        self.word_rnn_units = word_rnn_units
        self.word_rd = word_rd
        self.word_ed = word_ed
        self.word_dense = word_dense
        self.meta_rnn_units = meta_rnn_units
        self.meta_rd = meta_rd
        self.meta_ed = meta_ed
        self.meta_dense = meta_dense

    def init_word_model(self):
        input_layer = Input(shape=self.seq_length, name="word")
        embedding_layer = Embedding(
            self.vocab_size+1, self.embedding_size,
            input_length=self.seq_length,
            embeddings_initializer=self.embedding,
            mask_zero=True,
            trainable=False
        )(input_layer)
        embedding_layer2 = Embedding(
            self.vocab_size+1, self.embedding_size,
            input_length=self.seq_length,
            embeddings_initializer="glorot_normal",
            mask_zero=True,
            trainable=True
        )(input_layer)
        embedding_layer = Add()([embedding_layer, embedding_layer2])
        embedding_layer = Dropout(self.word_ed)(embedding_layer)
        lstm = Bidirectional(LSTM(
            self.word_rnn_units, return_sequences=True,
            recurrent_dropout=self.word_rd
        ))(embedding_layer)
        dense = Dense(self.word_dense, activation="relu", name="w_embed")(lstm)
        # out = Dense(self.n_label+1, name="w_out")(dense)
        # For classification
        word_block = Model(input_layer, dense)
        # For embedding (the one that concatenated for meta layer)
        # word_embed = Model(input_layer, dense)
        return word_block

    def init_char_model(self):
        input_layer = Input(shape=self.char_seq_length, name="char")
        embedding_layer = Embedding(
            self.n_chars+1, self.char_embedding_size,
            input_length=self.char_seq_length,
            embeddings_initializer="glorot_normal",
            mask_zero=True
        )(input_layer)
        embedding_layer = Dropout(self.char_ed)(embedding_layer)
        lstm = Bidirectional(LSTM(
            self.char_rnn_units, return_sequences=True,
            recurrent_dropout=self.char_rd
        ))(embedding_layer)
        # Convert Character embedding to word_embedding
        char_to_word = CharToWord(
            self.seq_length, self.char2idx["SEP"]
        )([lstm, input_layer])
        dense = Dense(self.char_dense, activation="relu", name="c_embed")(
            char_to_word
        )
        # For classification
        char_block = Model(input_layer, dense)
        # Alternative model
        # char_out = Input(shape=(self.char_length, self.char_dense))
        # out = Dense(self.n_label+1, name="c_out")(dense)
        # char_model = Model(char_out, out)
        # For embedding (the one that concatenated for meta layer)
        return char_block

    def init_meta_model(self):
        input_layer = Input(
            shape=(self.seq_length, self.char_dense+self.word_dense),
            name="meta"
        )
        lstm = Bidirectional(LSTM(
            self.meta_rnn_units, return_sequences=True,
            recurrent_dropout=self.meta_rd
        ))(input_layer)
        lstm = Dropout(self.meta_ed)(lstm)
        dense = Dense(self.meta_dense, activation="relu")(lstm)
        out = Dense(self.n_label+1)(dense)
        # For classification
        meta_block = Model(input_layer, out)
        return meta_block

    def init_model(self):
        # Char model
        char_block = self.init_char_model()
        # Word model
        word_block = self.init_word_model()
        # Meta model
        meta_block = self.init_meta_model()
        char_block.summary()
        word_block.summary()
        meta_block.summary()
        self.model = MetaModelWrapper(
            char_block, word_block, meta_block, self.n_label+1
        )
        self.model.compile(
            self.optimizer,
            self.optimizer,
            self.optimizer,
            CategoricalCrossentropy(from_logits=True)
        )

    def init_c2i(self):
        """
        Initialize character to index
        """
        vocab = set([*string.printable])
        self.char2idx = {ch: i+1 for i, ch in enumerate(vocab)}
        self.char2idx["UNK"] = len(self.char2idx)+1
        self.char2idx["SEP"] = len(self.char2idx)+1
        self.n_chars = len(self.char2idx)+1

    def get_char_vector(self, inp_seq):
        """
        Get character vector of the input sequence

        :param inp_seq: list of list of string, tokenized input corpus
        :return vector_seq: 3D numpy array, input vector on character level
        """
        vector_seq = np.zeros(
            (len(inp_seq), self.char_seq_length)
        )
        for i, data in enumerate(inp_seq):
            data = data[:self.seq_length]
            for j, word in enumerate(data):
                current_indexes = 1
                vector_seq[i, 0] = self.char2idx["SEP"]
                for k, ch in enumerate(word):
                    if ch in self.char2idx:
                        vector_seq[i, current_indexes] = self.char2idx[ch]
                    else:
                        vector_seq[i, current_indexes] = self.char2idx["UNK"]
                    current_indexes += 1
            vector_seq[i, current_indexes] = self.char2idx["SEP"]
        return vector_seq

    def vectorize_input(self, inp_seq):
        """
        Prepare vector of the input data

        :param inp_seq: list of list of string, tokenized input corpus
        :return input_vector: Dictionary, Word and char input vector
        """
        input_vector = {
            "word": self.get_word_vector(inp_seq),
            "char": self.get_char_vector(inp_seq)
        }
        return input_vector

    def init_inverse_indexes(self, X, y):
        super(MetaBiLSTMTagger, self).init_inverse_indexes(X, y)
        self.init_c2i()
