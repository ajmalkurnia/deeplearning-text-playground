from keras.layers import LSTM, Embedding, TimeDistributed, Concatenate, Add
from keras.layers import Dropout, Bidirectional, Dense
from keras.models import Model, Input
from keras.metrics import CategoricalAccuracy
from keras.losses import CategoricalCrossentropy
from model.AttentionText.attention_text import CharTagAttention
from model.base_tagger import BaseTagger


class StackedRNNTagger(BaseTagger):
    def __init__(
        self, word_length=50, char_embed_size=100,
        char_rnn_units=400, char_recurrent_dropout=0.33,
        recurrent_dropout=0.33, rnn_units=400, embedding_dropout=0.5,
        main_layer_dropout=0.5, **kwargs
    ):
        """
        Deep learning based sequence tagger.
        Consist of:
            - Char embedding (RNN+Attention)
            - RNN
        based on postag model of:
        https://www.aclweb.org/anthology/K17-3002/

        :param word_length: int, maximum character length in a token,
            relevant when using cnn
        :param char_embed_size: int, the size of character level embedding,
            relevant when using cnn
        :param char_rnn_units: int, RNN units on char level
        :param char_recurrent_dropout: float, dropout rate in RNN char level
        :param recurrent_dropout: float, dropout rate inside RNN
        :param rnn_units: int, the number of rnn units
        :param embedding_dropout: float, dropout rate after embedding layer
        :param main_layer_dropout: float, dropout rate in between LSTM
        """
        super(StackedRNNTagger, self).__init__(**kwargs)
        self.word_length = word_length
        self.char_embed_size = char_embed_size
        self.ed = embedding_dropout

        self.rnn_units = rnn_units
        self.rd = recurrent_dropout
        self.char_rnn_units = char_rnn_units
        self.char_rd = char_recurrent_dropout
        # self.loss =
        self.main_layer_dropout = main_layer_dropout

    def __get_char_embedding(self):
        """
        Initialize character embedding
        """
        word_input_layer = Input(shape=(self.word_length, ))
        # +1 for padding
        embedding_block = Embedding(
            self.n_chars+1, self.char_embed_size,
            input_length=self.word_length, trainable=True,
            mask_zero=True
        )(word_input_layer)

        rnn_output = LSTM(
            self.char_rnn_units, recurrent_dropout=self.char_rd,
            return_sequences=True, return_state=True
        )(embedding_block)
        embedding_block, h, c = rnn_output
        embedding_block = CharTagAttention(
            self.char_rnn_units, self.word_length
        )(embedding_block)
        embedding_block = Concatenate()([embedding_block, c])
        embedding_block = Dense(self.embedding_size)(embedding_block)

        embedding_block = Model(
            inputs=word_input_layer, outputs=embedding_block)
        embedding_block.summary()
        seq_inp_layer = Input(
            shape=(self.seq_length, self.word_length), name="char"
        )
        embedding_block = TimeDistributed(embedding_block)(seq_inp_layer)
        return seq_inp_layer, embedding_block

    def init_model(self):
        """
        Initialize the network model
        """
        # Word Embebedding
        input_word_layer = Input(shape=(self.seq_length,), name="word")
        pre_trained_word_embed = Embedding(
            self.vocab_size+1, self.embedding_size,
            input_length=self.seq_length,
            embeddings_initializer=self.embedding,
            mask_zero=True,
            trainable=False
        )
        pre_trained_word_embed = pre_trained_word_embed(input_word_layer)
        learnable_word_embed = Embedding(
            self.vocab_size+1, self.embedding_size,
            input_length=self.seq_length,
            embeddings_initializer="glorot_uniform",
            mask_zero=True
        )(input_word_layer)
        # Char Embedding
        input_char_layer, char_embed_block = self.__get_char_embedding()
        input_layer = [input_char_layer, input_word_layer]
        embed_block = Add()(
            [char_embed_block, pre_trained_word_embed, learnable_word_embed]
            )
        if self.ed > 0:
            embed_block = Dropout(self.ed)(embed_block)
        self.model = embed_block
        # LSTM Layer
        self.model = Bidirectional(LSTM(
            self.rnn_units, return_sequences=True,
            recurrent_dropout=self.rd
        ))(self.model)
        self.model = Dropout(self.main_layer_dropout)(self.model)
        self.model = Bidirectional(LSTM(
            self.rnn_units, return_sequences=True,
            recurrent_dropout=self.rd
        ))(self.model)
        # Dense layer
        self.model = Dense(self.n_label+1, activation="relu")(self.model)
        out = Dense(self.n_label+1)(self.model)
        # out = TimeDistributed(
        #     Dense(self.n_label+1, activation="softmax")
        # )(self.model)

        self.model = Model(input_layer, out)
        self.model.summary()
        self.model.compile(
            # Compute loss from logits because the output is not probability
            loss=CategoricalCrossentropy(from_logits=True),
            # use CategoricalAccuracy so it can operates using logits
            optimizer=self.optimizer, metrics=[CategoricalAccuracy()]
        )

    def vectorize_input(self, inp_seq):
        """
        Prepare vector of the input data

        :param inp_seq: list of list of string, tokenized input corpus
        :return word_vector: Dictionary, Word and char input vector
        """
        input_vector = {
            "word": self.get_word_vector(inp_seq),
            "char": self.get_char_vector(inp_seq)
        }
        return input_vector

    def init_inverse_indexes(self, X, y):
        super(StackedRNNTagger, self).init_inverse_indexes(X, y)
        self.init_c2i()

    def get_class_param(self):
        class_param = {
            "label2idx": self.label2idx,
            "word2idx": self.word2idx,
            "seq_length": self.seq_length,
            "word_length": self.word_length,
            "idx2label": self.idx2label,
            "char2idx": self.char2idx
        }
        return class_param

    @staticmethod
    def init_from_config(class_param):
        """
        Load model from the saved zipfile

        :param filepath: path to model zip file
        :return classifier: Loaded model class
        """
        constructor_param = {
            "seq_length": class_param["seq_length"],
            "word_length": class_param["word_length"],
        }
        classifier = StackedRNNTagger(**constructor_param)
        classifier.label2idx = class_param["label2idx"]
        classifier.word2idx = class_param["word2idx"]
        classifier.idx2label = class_param["idx2label"]
        classifier.n_label = len(classifier.label2idx)
        classifier.char2idx = class_param["char2idx"]
        return classifier
