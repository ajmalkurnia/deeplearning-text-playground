from keras.layers import LSTM, Embedding, TimeDistributed, Concatenate, Add
from keras.layers import Dropout, Bidirectional, Dense
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input

import numpy as np
import string

from model.AttentionText.attention_text import CharTagAttention
from model.base_tagger import BaseTagger


class StackedRNNTagger(BaseTagger):
    def __init__(
        self, word_length=50, char_embed_size=100,
        char_rnn_units=25, char_recurrent_dropout=0.33,
        recurrent_dropout=0.33, rnn_units=400, embedding_dropout=0.5,
        main_layer_dropout=0.5, **kwargs
    ):
        """
        Deep learning based sequence tagger.
        Consist of:
            - Char embedding (CNN Optional)
            - RNN
            - CRF (Optional)

        :param word_length: int, maximum character length in a token,
            relevant when using cnn
        :param char_embed_size: int, the size of character level embedding,
            relevant when using cnn
        :param recurrent_dropout: float, dropout rate inside RNN
        :param embedding_dropout: float, dropout rate after embedding layer
        :param rnn_units: int, the number of rnn units
        :param pre_outlayer_dropout: float, dropout rate before output layer
        :param char_embed_type: bool, whether to use cnn as character embedding
        :param crf: bool, whether to use crf or softmax
        :param conv_layers: list of list, convolution layer settings,
            relevant when using cnn char embedding
            each list component consist of 3 length tuple/list that denotes:
                int, number of filter,
                int, filter size,
                int, maxpool size (use -1 to not use maxpooling)
            each convolution layer is connected directly to embedding layer,
            character information will be obtained by applying concatenation
                and GlobalMaxPooling
        """
        super(StackedRNNTagger, self).__init__(**kwargs)
        self.word_length = word_length
        self.char_embed_size = char_embed_size
        self.ed = embedding_dropout

        self.rnn_units = rnn_units
        self.rd = recurrent_dropout
        self.char_rnn_units = char_rnn_units
        self.char_rd = char_recurrent_dropout

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
            self.char_rnn_unit, recurrent_dropout=self.char_rd,
            return_sequences=True, return_state=True
        )(embedding_block)
        embedding_block, h, c = rnn_output
        embedding_block = CharTagAttention(
            self.char_embed_size, self.word_length
        )(embedding_block)
        embedding_block = Concatenate()([embedding_block, c])
        embedding_block = Dense(self.word_embed_size)(embedding_block)

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
            self.vocab_size+1, self.word_embed_size,
            input_length=self.seq_length,
            embeddings_initializer=self.embedding,
            mask_zero=True,
            trainable=False
        )
        pre_trained_word_embed = pre_trained_word_embed(input_word_layer)
        learnable_word_embed = Embedding(
            self.vocab_size+1, self.word_embed_size,
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
        self.model = Bidirectional(LSTM(
            units=self.rnn_units, return_sequences=True,
            dropout=self.rd,
        ))(self.model)
        self.model = Dropout(self.main_layer_dropout)(self.model)
        self.model = Bidirectional(LSTM(
            units=self.rnn_units, return_sequences=True,
            dropout=self.rd,
        ))(self.model)
        self.model = Dense(self.n_label+1, activation="relu")(self.model)
        # Dense layer
        out = Dense(self.n_label+1)(self.model)
        self.model = Model(input_layer, out)
        self.model.summary()
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def __init_transition_matrix(self, y):
        n_labels = self.n_label + 1
        self.transition_matrix = np.zeros((n_labels, n_labels))
        for data in y:
            for idx, label in enumerate(data[:self.seq_length]):
                if idx:
                    current = self.label2idx[label]
                    prev = self.label2idx[data[idx-1]]
                    self.transition_matrix[prev][current] += 1
            self.transition_matrix[current][0] += 1
            zero_pad = self.seq_length - len(data)
            if zero_pad > 1:
                self.transition_matrix[0][0] += zero_pad
        for row in self.transition_matrix:
            s = sum(row)
            row[:] = (row + 1)/(s+n_labels)
        return self.transition_matrix

    def init_c2i(self):
        """
        Initialize character to index
        """
        vocab = set([*string.printable])
        self.char2idx = {ch: i+1 for i, ch in enumerate(vocab)}
        self.char2idx["UNK"] = len(self.char2idx)+1
        self.n_chars = len(self.char2idx)+1

    def get_char_vector(self, inp_seq):
        """
        Get character vector of the input sequence
        :param inp_seq: list of list of string, tokenized input corpus
        :return vector_seq: 3D numpy array, input vector on character level
        """
        vector_seq = np.zeros(
            (len(inp_seq), self.seq_length, self.word_length)
        )
        for i, data in enumerate(inp_seq):
            data = data[:self.seq_length]
            for j, word in enumerate(data):
                word = word[:self.word_length]
                for k, ch in enumerate(word):
                    if ch in self.char2idx:
                        vector_seq[i, j, k] = self.char2idx[ch]
                    else:
                        vector_seq[i, j, k] = self.char2idx["UNK"]
        return vector_seq

    def vectorize_input(self, inp_seq):
        """
        Prepare vector of the input data
        :param inp_seq: list of list of string, tokenized input corpus
        :return word_vector: 2D numpy array, input vector on word level
        :return char_vector: 3D numpy array, input vector on character level
            return None when not using any char_embedding
        """
        input_vector = {}
        input_vector["word"] = self.get_word_vector(inp_seq)
        input_vector["char"] = self.get_char_vector(inp_seq)
        return input_vector

    def vectorize_label(self, out_seq):
        """
        Get prepare vector of the label for training
        :param out_seq: list of list of string, tokenized input corpus
        :return out_seq: 2D/3D numpy array, vector of label data
            return 2D array when using crf
            return 3D array when not using crf
        """
        out_seq = [[self.label2idx[w] for w in s] for s in out_seq]
        out_seq = pad_sequences(
            maxlen=self.seq_length, sequences=out_seq, padding="post"
        )
        out_seq = [
            to_categorical(i, num_classes=self.n_label+1) for i in out_seq
        ]
        return np.array(out_seq)

    def devectorize_label(self, pred_sequence, input_data):
        """
        Get readable label sequence
        :param pred_sequence: 4 length list, prediction results from CRF layer
        :param input_data: list of list of string, tokenized input corpus
        :return label_seq: list of list of string, readable label sequence
        """
        return self.get_greedy_label(pred_sequence, input_data)

    def init_training(self, X, y):
        self.init_l2i(y)
        if self.word2idx is None:
            self.init_w2i(X)
        self.init_embedding()
        self.init_c2i()
        self.init_model()

    def get_class_param(self, filepath, zipf):
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
