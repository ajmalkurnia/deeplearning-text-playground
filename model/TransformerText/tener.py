from keras.layers import Dense, Dropout, Embedding, Input, TimeDistributed
from keras.layers import GlobalMaxPooling1D, Concatenate
from keras.initializers import Constant, RandomUniform
from keras.models import Model
from tensorflow_addons.layers.crf import CRF
import numpy as np

from model.extras.crf_subclass_model import ModelWithCRFLoss
from model.base_crf_out_tagger import BaseCRFTagger
from model.TransformerText.relative_transformer_block import TransformerBlock


class TENERTagger(BaseCRFTagger):
    def __init__(
        self, word_length=50, char_embed_size=30, char_heads=3,
        char_dim_ff=60, n_blocks=2, dim_ff=128, n_heads=6,
        attention_dim=256, fcn_layers=[(512, 0.3, "relu")],
        transformer_dropout=0.3, attention_dropout=0.5, embedding_dropout=0.5,
        out_transformer_dropout=0.3, scale=False, **kwargs
    ):
        """
        TENER tagger based on:
        https://arxiv.org/abs/1911.04474
        https://github.com/fastnlp/TENER
        (note. still not sure if this is accurate)

        :param word_length: int, number of characters in a word
        :param char_embed_size: int, character embedding size
        :param char_heads: int, number of attention head on char level
        :param char_dim_ff: int, ff unit for char level transformer
        :param n_blocks: int, number of transformer stack
        :param dim_ff: int, hidden unit on fcn layer in transformer
        :param n_heads: int, number of attention heads
        :param attention_dim: int, number of unit for transformer in and out
        :param fcn_layers: 2D list-like, Fully Connected layer settings,
            each list element denotes config for 1 layer,
                each config consist of 3 length tuple/list that denotes:
                    int, number of dense unit
                    float, dropout after dense layer
                    activation, activation function
        :param transformer_dropout: float, dropout rate inside transformer
        :param attention_dropout: float, dropout rate on multi head attention
        :param embedding_dropout: float, dropout rate after embedding
        :param out_transformer_dropout: float, dropout rate after
            transformer block
        :param scale: float, attention scaling 1/None
        """
        # self.__doc__ = BaseClassifier.__doc__
        super(TENERTagger, self).__init__(**kwargs)
        self.n_blocks = n_blocks
        self.dim_ff = dim_ff
        self.td = transformer_dropout
        self.ed = embedding_dropout
        self.ad = attention_dropout
        self.otd = out_transformer_dropout
        self.n_heads = n_heads
        self.attention_dim = attention_dim
        self.scale = scale
        self.fcn_layers = fcn_layers
        self.word_length = word_length
        self.char_embed_size = char_embed_size
        self.char_heads = char_heads
        self.char_dim_ff = char_dim_ff
        self.loss = "sparse_categorical_crossentropy"

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
        embedding_block = TransformerBlock(
            self.char_dim_ff, self.char_heads, self.char_embed_size, self.td,
            self.ad, self.scale
        )(embedding_block)
        embedding_block = Dense(self.char_embed_size)(embedding_block)
        embedding_block = GlobalMaxPooling1D()(embedding_block)
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

    def init_model(self):
        """
        Initialize TENER model architecture
        """
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
        input_char_layer, char_embed_block = self.__get_char_embedding()
        input_layer = [input_char_layer, input_word_layer]
        embed_block = Concatenate()([char_embed_block, word_embed_block])
        self.model = Dense(self.attention_dim)(embed_block)
        for _ in range(self.n_blocks):
            self.model = TransformerBlock(
                self.dim_ff, self.n_heads, self.attention_dim, self.td,
                self.ad, self.scale
            )(self.model)
        self.model = Dropout(self.otd)(self.model)
        for units, do_rate, activation in self.fcn_layers:
            self.model = Dense(units, activation=activation)(self.model)
            self.model = Dropout(do_rate)(self.model)
        crf = CRF(
            self.n_label+1,
            chain_initializer=Constant(self.transition_matrix)
        )
        out = crf(self.model)
        self.model = Model(inputs=input_layer, outputs=out)
        self.model.summary()
        # Subclassing to properly compute crf loss
        self.model = ModelWithCRFLoss(self.model)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

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
        super(TENERTagger, self).init_inverse_indexes(X, y)
        self.init_c2i()

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
            "word_length": class_param["word_length"]
        }
        classifier = TENERTagger(**constructor_param)
        classifier.label2idx = class_param["label2idx"]
        classifier.word2idx = class_param["word2idx"]
        classifier.idx2label = class_param["idx2label"]
        classifier.n_label = len(classifier.label2idx)
        classifier.char2idx = class_param["char2idx"]
        return classifier
