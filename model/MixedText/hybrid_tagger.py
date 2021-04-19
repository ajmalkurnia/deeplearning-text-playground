from keras.layers import LSTM, Embedding, TimeDistributed, Concatenate
from keras.layers import Dropout, Bidirectional, Conv1D, MaxPooling1D
from keras.layers import GlobalMaxPooling1D, Dense
from keras.utils import to_categorical
from keras.initializers import Constant, RandomUniform
from keras.models import Model, Input
from tensorflow_addons.layers.crf import CRF
import numpy as np

from model.extras.crf_subclass_model import ModelWithCRFLoss
from model.TransformerText.relative_transformer_block import TransformerBlock
from model.base_crf_out_tagger import BaseCRFTagger


class DLHybridTagger(BaseCRFTagger):
    def __init__(
        self, word_length=50, char_embed_type="cnn", char_embed_size=30,
        main_layer_type="rnn",
        # char convolution config
        char_conv_config=[[30, 3, -1], [30, 2, -1], [30, 4, -1]],
        # char transformer config
        char_trans_block=1, char_trans_head=3, char_trans_dim_ff=60,
        char_trans_dropout=0.3, char_attention_dropout=0.5, char_trans_scale=1,
        char_rnn_units=25, char_recurrent_dropout=0.33,
        # transformer config
        trans_blocks=2, trans_heads=8, trans_dim_ff=256, trans_dropout=0.5,
        attention_dropout=0.5, trans_scale=1, trans_attention_dim=256,
        # rnn config
        recurrent_dropout=0.5, rnn_units=100, embedding_dropout=0.5,
        main_layer_dropout=0.5, fcn_layers=[],
        use_crf=True, **kwargs
    ):
        """
        Hybrid Deep learning based sequence tagger.
        Consist of:
            - Char block [CNN, RNN, AdaTransformer (TENER)](Optional)
            - Main block [RNN, AdaTransformer (TENER)]
            - Output block [Softmax, CRF]

        CNN-RNN-CRF based on:
            https://www.aclweb.org/anthology/P16-1101/
        TENER-TENER-CRF based on:
            https://arxiv.org/abs/1911.04474
        RNN-RNN-CRF based on:
            https://www.aclweb.org/anthology/N16-1030/

        :param word_length: int, maximum character length in a token,
            relevant when using char level embedding
        :param char_embed_type: string, method to get char embedding
            available option, cnn/rnn/adatrans
            supply None to not use char embedding
        :param char_embed_size: int, the size of character level embedding,
            relevant when using char level embedding
        :param main_layer_type: string, method on main layer (block)
            available option, rnn/adatrans
        :param conv_layers: list of list, convolution layer settings,
            relevant when using cnn char embedding
            each list component consist of 3 length tuple/list that denotes:
                int, number of filter,
                int, filter size,
                int, maxpool size (use -1 to not use maxpooling)
            each convolution layer is connected directly to embedding layer,
            character information will be obtained by applying concatenation
                and GlobalMaxPooling
        :param char_trans_block: int, number of transformer block on char level
        :param char_trans_head: int, number of attention head on char level
        :param char_trans_dim_ff: int, ff units in char level transformer
        :param char_trans_dropout: float, dropout rate in char level
            transformer
        :param char_attention_dropout: float, dropout rate for attention on
            char level transformer
        :param char_trans_scale: int, attention scaling on char level
            transformer
        :param char_rnn_units: int, rnn units on char level
        :param char_recurrent_dropout: float, dropout rate in char level RNN
        :param trans_block: int, number of transformer block for main layer
        :param trans_head: int, number of attention head for main layer
        :param trans_dim_ff: int, ff units in main layer transformer
        :param trans_dropout: float, dropout rate in main layer transformer
        :param attention_dropout: float, dropout rate for attention on
            main layer transformer
        :param trans_scale: int, attention scaling on main layer transformer
        :param trans_attention_dim: int, attention dimension for main layer
            transformer
        :param rnn_units: int, number of rnn units
        :param recurrent_dropout: float, dropout rate in RNN
        :param embedding_dropout: float, dropout rate after embedding layer
        :param main_layer_dropout: float, dropout rate after main layer
        :param fcn_layers: 2D list-like, Fully Connected layer settings,
            will be placed after conv layer,
            each list element denotes config for 1 layer,
                each config consist of 3 length tuple/list that denotes:
                    int, number of dense unit
                    float, dropout after dense layer
                    activation, activation function
        :param crf: bool, whether to use crf or softmax
        """
        super(DLHybridTagger, self).__init__(**kwargs)
        self.word_length = word_length
        self.char_embed_size = char_embed_size
        self.ed = embedding_dropout

        self.rnn_units = rnn_units
        self.rd = recurrent_dropout
        self.char_rnn_units = char_rnn_units
        self.char_rd = char_recurrent_dropout

        self.main_layer_dropout = main_layer_dropout
        self.use_crf = use_crf
        self.char_embed_type = char_embed_type
        self.char_conv_config = char_conv_config

        self.char_trans_block = char_trans_block
        self.char_trans_head = char_trans_head
        self.char_trans_dim_ff = char_trans_dim_ff
        self.char_trans_dropout = char_trans_dropout
        self.char_attention_dropout = char_attention_dropout
        self.char_trans_scale = char_trans_scale

        self.trans_blocks = trans_blocks
        self.trans_heads = trans_heads
        self.trans_dim_ff = trans_dim_ff
        self.trans_attention_dim = trans_attention_dim
        self.td = trans_dropout
        self.ad = attention_dropout
        self.trans_scale = trans_scale

        self.fcn_layers = fcn_layers
        self.main_layer_type = main_layer_type
        if self.use_crf:
            self.loss = "sparse_categorical_crossentropy"

    def char_cnn_block(self, embedding_block):
        """
        Initialize character level CNN
        :param embedding_block: Embedding layer, char embedding layer
        :return embedding_block: keras.Layer, char embedding block
        """
        conv_layers = []
        for filter_num, filter_size, pooling_size in self.char_conv_config:
            conv_layer = Conv1D(
                filter_num, filter_size, activation="relu"
            )(embedding_block)
            if pooling_size != -1:
                conv_layer = MaxPooling1D(
                    pool_size=pooling_size
                )(conv_layer)
            conv_layers.append(conv_layer)
        embedding_block = Concatenate(axis=1)(conv_layers)
        embedding_block = GlobalMaxPooling1D()(embedding_block)
        return embedding_block

    def char_rnn_block(self, embedding_block):
        """
        Initialize character level RNN
        :param embedding_block: Embedding layer, char embedding layer
        :return embedding_block: keras.Layer, char embedding block
        """
        return Bidirectional(
            LSTM(self.char_rnn_units, recurrent_dropout=self.char_rd)
        )(embedding_block)

    def char_trans_block(self, embedding_block):
        """
        Initialize character level Adaptive Transformer (TENER)
        :param embedding_block: Embedding layer, char embedding layer
        :return embedding_block: keras.Layer, char embedding block
        """
        for _ in range(self.char_trans_block):
            embedding_block = TransformerBlock(
                self.char_trans_dim_ff, self.char_trans_head,
                self.char_embed_size, self.char_trans_dropout,
                self.char_attention_dropout, self.char_trans_scale
            )(embedding_block)
        embedding_block = Dense(self.char_embed_size)(embedding_block)
        embedding_block = GlobalMaxPooling1D()(embedding_block)
        return embedding_block

    def __get_char_embedding(self):
        """
        Initialize character embedding
        """
        word_input_layer = Input(shape=(self.word_length, ))
        # +1 for padding
        embedding_block = Embedding(
            self.n_chars+1, self.char_embed_size,
            input_length=self.word_length, trainable=True,
            mask_zero=True,
            embeddings_initializer=RandomUniform(
                minval=-1*np.sqrt(3/self.char_embed_size),
                maxval=np.sqrt(3/self.char_embed_size)
            )
        )(word_input_layer)
        if self.char_embed_type == "cnn":
            embedding_block = self.char_cnn_block(embedding_block)
        elif self.char_embed_type == "adatrans":
            embedding_block = self.char_trans_block(embedding_block)
        elif self.char_embed_type == "rnn":
            embedding_block = self.char_rnn_block(embedding_block)
        else:
            raise ValueError("Invalid char embedding type")

        if self.ed > 0:
            embedding_block = Dropout(self.ed)(embedding_block)
        embedding_block = Model(
            inputs=word_input_layer, outputs=embedding_block)
        embedding_block.summary()
        seq_inp_layer = Input(
            shape=(self.seq_length, self.word_length), name="char"
        )
        embedding_block = TimeDistributed(embedding_block)(seq_inp_layer)
        return seq_inp_layer, embedding_block

    def main_rnn_block(self, embed_block):
        """
        Initialize RNN main layer
        :param embed_block: embedding layer, word/concat word+char
        :param main_layer: Bidirectional LSTM layer
        """
        main_layer = Bidirectional(LSTM(
            units=self.rnn_units, return_sequences=True,
            recurrent_dropout=self.rd,
        ))(embed_block)
        return main_layer

    def main_trans_block(self, embed_block):
        """
        Initialize Adaptive Transformer (TENER) main layer
        :param embed_block: embedding layer, word/concat word+char
        :param main_layer: TENER
        """
        main_layer = Dense(self.trans_attention_dim)(embed_block)
        for _ in range(self.trans_blocks):
            main_layer = TransformerBlock(
                self.trans_dim_ff, self.trans_heads, self.trans_attention_dim,
                self.td, self.ad, self.trans_scale
            )(main_layer)
        return main_layer

    def init_model(self):
        """
        Initialize the network model
        """
        # Word Embebedding
        input_word_layer = Input(shape=(self.seq_length,), name="word")
        word_embed_block = Embedding(
            self.vocab_size+1, self.word_embed_size,
            input_length=self.seq_length,
            embeddings_initializer=self.embedding,
            mask_zero=True,
        )
        word_embed_block = word_embed_block(input_word_layer)
        if self.ed > 0:
            word_embed_block = Dropout(self.ed)(word_embed_block)
        # Char Embedding
        if self.char_embed_type:
            input_char_layer, char_embed_block = self.__get_char_embedding()
            input_layer = [input_char_layer, input_word_layer]
            embed_block = Concatenate()([char_embed_block, word_embed_block])
        else:
            embed_block = word_embed_block
            input_layer = input_word_layer
        # Main layer
        if self.main_layer_type == "rnn":
            self.model = self.main_rnn_block(embed_block)
        elif self.main_layer_type == "adatrans":
            self.model = self.main_trans_block(embed_block)
        # FCN layer
        self.model = Dropout(self.main_layer_dropout)(self.model)
        for units, do_rate, activation in self.fcn_layers:
            self.model = Dense(units, activation=activation)(self.model)
            self.model = Dropout(do_rate)(self.model)
        # Output Layer
        if self.use_crf:
            crf = CRF(
                self.n_label+1,
                chain_initializer=Constant(self.transition_matrix)
            )
            out = crf(self.model)
            self.model = Model(inputs=input_layer, outputs=out)
            self.model.summary()
            # Subclassing to properly compute crf loss
            self.model = ModelWithCRFLoss(self.model)
        else:
            # Dense layer
            out = TimeDistributed(Dense(
                self.n_label+1, activation="softmax"
            ))(self.model)
            self.model = Model(input_layer, out)
            self.model.summary()
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def vectorize_input(self, inp_seq):
        """
        Prepare vector of the input data
        :param inp_seq: list of list of string, tokenized input corpus
        :return input_vector: Dictionary/np.array, Word and char input vector
            return dictionary when using character embedding
            return np.array when not using character embedding
        """
        input_vector = {}
        input_vector["word"] = self.get_word_vector(inp_seq)
        if self.char_embed_type:
            input_vector["char"] = self.get_char_vector(inp_seq)
        else:
            input_vector = input_vector["word"]
        return input_vector

    def vectorize_label(self, out_seq):
        """
        Get prepare vector of the label for training
        :param out_seq: list of list of string, tokenized input corpus
        :return out_seq: 2D/3D numpy array, vector of label data
            return 2D array when using crf
            return 3D array when not using crf
        """
        out_seq = super(DLHybridTagger, self).vectorize_label(out_seq)
        if not self.use_crf:
            # the label for Dense output layer needed to be onehot encoded
            out_seq = [
                to_categorical(i, num_classes=self.n_label+1) for i in out_seq
            ]
        return np.array(out_seq)

    def devectorize_label(self, pred_sequence, input_data):
        """
        Get readable label sequence
        :param pred_sequence: np.array/list, prediction results from out layer
            np.array when using softmax
            list when using CRF
        :param input_data: list of list of string, tokenized input corpus
        :return label_seq: list of list of string, readable label sequence
        """
        if self.use_crf:
            return self.get_crf_label(pred_sequence, input_data)
        else:
            return self.get_greedy_label(pred_sequence, input_data)

    def init_inverse_indexes(self, X, y):
        super(DLHybridTagger, self).init_inverse_indexes(X, y)
        if self.char_embed_type:
            self.init_c2i()

    def training_prep(self, X, y):
        """
        Initialized necessary class attributes for training

        :param X: 2D list, training dataset in form of tokenized corpus
        :param y: 2D list, training data label
        """
        self.init_inverse_indexes(X, y)
        if self.use_crf:
            self.compute_transition_matrix(y)
        self.init_embedding()
        self.init_model()

    def save_network(self, filepath, zipf):
        """
        Saving keras model

        :param filepath: string, save file path
        :param zipf: zipfile
        """
        if self.use_crf:
            super(DLHybridTagger, self).save_network(filepath, zipf)
        else:
            super(BaseCRFTagger, self).save_network(filepath, zipf)

    def get_class_param(self):
        class_param = {
            "label2idx": self.label2idx,
            "word2idx": self.word2idx,
            "seq_length": self.seq_length,
            "word_length": self.word_length,
            "idx2label": self.idx2label,
            "use_crf": self.use_crf,
            "char_embed_type": self.char_embed_type
        }
        if self.char_embed_type:
            class_param["char2idx"] = self.char2idx
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
            "use_crf": class_param["use_crf"],
            "char_embed_type": class_param["char_embed_type"]
        }
        classifier = DLHybridTagger(**constructor_param)
        classifier.label2idx = class_param["label2idx"]
        classifier.word2idx = class_param["word2idx"]
        classifier.idx2label = class_param["idx2label"]
        classifier.n_label = len(classifier.label2idx)
        if "char2idx" in class_param:
            classifier.char2idx = class_param["char2idx"]
        return classifier
