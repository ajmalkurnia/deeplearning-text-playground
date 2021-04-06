from keras.layers import LSTM, Embedding, TimeDistributed, Concatenate
from keras.layers import Dropout, Bidirectional, Conv1D, MaxPooling1D
from keras.layers import GlobalMaxPool1D, Dense
from keras.utils import to_categorical
from keras.initializers import Constant, RandomUniform
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input, load_model
from keras.callbacks import EarlyStopping
from tensorflow_addons.layers.crf import CRF

from common.word_vector import WE_TYPE
from .tf_model_crf import ModelWithCRFLoss

from tempfile import TemporaryDirectory
from collections import defaultdict
from zipfile import ZipFile
import numpy as np
import pickle
import string
import os


class DLHybridTagger():
    def __init__(
        self, seq_length=100, word_length=50, char_embed_size=30,
        word_embed_size=100, word_embed_file=None, we_type="glorot_normal",
        recurrent_dropout=0.5, embedding_dropout=0.5, rnn_units=100,
        optimizer="adam", loss=None, vocab_size=10000,
        pre_outlayer_dropout=0.5, char_embedding="cnn", crf=True,
        conv_layers=[[30, 3, -1], [30, 2, -1], [30, 4, -1]],
    ):
        """
        Deep learning based sequence tagger.
        Consist of:
            - Char embedding (CNN Optional)
            - RNN
            - CRF (Optional)

        :param seq_length: int, maximum sequence length in a data
        :param word_length: int, maximum character length in a token,
            relevant when char_embedding is not None
        :param char_embed_size: int, the size of character level embedding,
            relevant when char_embedding is not None
        :param word_embed_size: int, the size of word level embedding,
            relevant when not using pretrained embedding file
        :param word_embed_file: string, path to pretrained word embedding
        :param we_type: string, word embedding types:
            random, supply any keras initilaizer string
            pretrained, word embedding type of the word_embed_file,
                available option: "w2v", "ft", "glove"
        :param recurrent_dropout: float, dropout rate inside RNN
        :param embedding_dropout: float, dropout rate after embedding layer
        :param rnn_units: int, the number of rnn units
        :param optimizer: string/object, any valid optimizer parameter
            during model compilation
        :param loss: string/object, any valid loss parameter
            during model compilation
        :param vocab_size: int, the size of vobulary for the embedding
        :param pre_outlayer_dropout: float, dropout rate before output layer
        :param char_embedding: string/none, the type of character embedding
            valid option:
            - "cnn" to use cnn based character embedding
            - None to not use any character embedding
        :param crf: bool, using CRF as output layer,
            if false time distributed softmax layer will be used
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

        self.seq_length = seq_length
        self.word_length = word_length

        self.char_embed_size = char_embed_size
        # Will be overide with pretrained file embedding
        self.word_embed_size = word_embed_size
        self.word_embed_file = word_embed_file
        self.we_type = we_type

        if we_type in ["w2v", "ft", "glove"] and word_embed_file is None:
            raise ValueError(
                "Supply parameter word_embed_file when using w2v/ft/glove embedding"  # noqa
            )

        self.rnn_units = rnn_units
        self.rd = recurrent_dropout
        self.ed = embedding_dropout

        self.pre_outlayer_dropout = pre_outlayer_dropout
        self.crf = crf

        self.char_embedding = char_embedding
        if self.char_embedding not in ["cnn", None]:
            raise ValueError(
                "Invalid character embedding, valid value: 'cnn' or None"
            )

        self.conv_layers = conv_layers

        if loss is None and self.crf:
            self.loss = "sparse_categorical_crossentropy"
        elif loss is None:
            self.loss = "categorical_crossentropy"
        else:
            self.loss = loss

        self.optimizer = optimizer
        self.vocab_size = vocab_size

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
        conv_layers = []
        for filter_num, filter_size, pooling_size in self.conv_layers:
            conv_layer = Conv1D(
                filter_num, filter_size, activation="relu"
            )(embedding_block)
            if pooling_size != -1:
                conv_layer = MaxPooling1D(
                    pool_size=pooling_size
                )(conv_layer)
            conv_layers.append(conv_layer)
        embedding_block = Concatenate(axis=1)(conv_layers)
        embedding_block = GlobalMaxPool1D()(embedding_block)
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

    def __init_model(self, y):
        """
        Initialize the network model
        """
        # Word Embebedding
        input_word_layer = Input(shape=(self.seq_length,), name="word")
        word_embed_block = Embedding(
            self.vocab_size+1, self.word_embed_size,
            input_length=self.seq_length,
            embeddings_initializer=self.word_embedding,
            mask_zero=True,
        )
        word_embed_block = word_embed_block(input_word_layer)
        if self.ed > 0:
            word_embed_block = Dropout(self.ed)(word_embed_block)
        # Char Embedding
        if self.char_embedding == "cnn":
            input_char_layer, char_embed_block = self.__get_char_embedding()
            input_layer = [input_char_layer, input_word_layer]
            embed_block = Concatenate()([char_embed_block, word_embed_block])
        else:
            embed_block = word_embed_block
            input_layer = input_word_layer
        # RNN
        self.model = Bidirectional(LSTM(
            units=self.rnn_units, return_sequences=True,
            dropout=self.rd,
        ))(embed_block)
        self.model = Dropout(self.pre_outlayer_dropout)(self.model)
        if self.crf:
            crf = CRF(
                self.n_label+1,
                chain_initializer=Constant(self.__compute_transition_matrix(y))
            )
            out = crf(self.model)
            self.model = Model(
                inputs=input_layer, outputs=out
            )
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
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer
        )

    def __compute_transition_matrix(self, y):
        transition_matrix = np.zeros((self.n_label+1, self.n_label+1))
        for data in y:
            for idx, label in enumerate(data[:self.seq_length]):
                if idx:
                    current = self.label2idx[label]
                    prev = self.label2idx[data[idx-1]]
                    transition_matrix[prev][current] += 1
            transition_matrix[current][0] += 1
            zero_pad = self.seq_length - len(data)
            if zero_pad > 1:
                transition_matrix[0][0] += zero_pad
        for row in transition_matrix:
            s = sum(row)
            row[:] = (row + 1)/(s+self.n_label+1)
        return transition_matrix

    def __init_c2i(self):
        """
        Initialize character to index
        """
        vocab = set([*string.printable])
        self.char2idx = {ch: i+1 for i, ch in enumerate(vocab)}
        self.char2idx["UNK"] = len(self.char2idx)+1
        self.n_chars = len(self.char2idx)+1

    def __init_w2i(self, data):
        """
        Initialize word to index
        """
        vocab = defaultdict(int)
        for s in data:
            for w in s:
                vocab[w] += 1
        vocab = sorted(vocab.items(), key=lambda d: (d[1], d[0]))
        vocab = [v[0] for v in vocab]
        # +1 Padding +1 UNK
        vocab = list(reversed(vocab))[:self.vocab_size-2]
        self.word2idx = {word: idx+1 for idx, word in enumerate(vocab)}
        self.word2idx["[UNK]"] = len(self.word2idx)+1
        self.n_words = len(self.word2idx)+1

    def __init_l2i(self, data):
        """
        Initialize label to index
        """
        label = list(set([lb for sub in data for lb in sub]))
        self.n_label = len(label)
        self.label2idx = {ch: idx+1 for idx, ch in enumerate(sorted(label))}
        self.idx2label = {idx: ch for ch, idx in self.label2idx.items()}

    def __init_wv_embedding(self):
        """
        Initialization of for Word embedding matrix
        UNK word will be initialized randomly
        """
        wv_model = WE_TYPE[self.we_type].load_model(self.word_embed_file)
        self.word_embed_size = wv_model.size

        self.word_embedding = np.zeros(
            (self.vocab_size+1, wv_model.size), dtype=float
        )
        for word, idx in self.word2idx.items():
            self.word_embedding[idx, :] = wv_model.retrieve_vector(word)

    def __init_embedding(self):
        """
        Initialize argument for word embedding initializer
        """
        if self.we_type in ["w2v", "ft", "glove"]:
            self.__init_wv_embedding()
            self.word_embedding = Constant(self.word_embedding)
        else:
            self.word_embedding = self.we_type

    def __char_vector(self, inp_seq):
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

    def __word_vector(self, inp_seq):
        """
        Get word vector of the input sequence
        :param inp_seq: list of list of string, tokenized input corpus
        :return vector_seq: 2D numpy array, input vector on word level
        """
        vector_seq = np.zeros((len(inp_seq), self.seq_length))
        for i, data in enumerate(inp_seq):
            data = data[:self.seq_length]
            for j, word in enumerate(data):
                if word in self.word2idx:
                    vector_seq[i][j] = self.word2idx[word]
                else:
                    vector_seq[i][j] = self.word2idx["[UNK]"]
        return vector_seq

    def vectorize_input(self, inp_seq):
        """
        Prepare vector of the input data
        :param inp_seq: list of list of string, tokenized input corpus
        :return word_vector: 2D numpy array, input vector on word level
        :return char_vector: 3D numpy array, input vector on character level
            return None when not using any char_embedding
        """
        word_vector = self.__word_vector(inp_seq)
        if self.char_embedding:
            char_vector = self.__char_vector(inp_seq)
        else:
            char_vector = None
        return word_vector, char_vector

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
        if not self.crf:
            # the label for Dense output layer needed to be onehot encoded
            out_seq = [
                to_categorical(i, num_classes=self.n_label+1) for i in out_seq
            ]
        return np.array(out_seq)

    def get_crf_label(self, pred_sequence, input_data):
        """
        Get label sequence
        :param pred_sequence: 4 length list, prediction results from CRF layer
        :param input_data: list of list of string, tokenized input corpus
        :return label_seq: list of list of string, readable label sequence
        """
        label_seq = []
        for i, s in enumerate(pred_sequence[0]):
            tmp = []
            for j, w in enumerate(s[:len(input_data[i])]):
                if w in self.idx2label:
                    label = self.idx2label[w]
                else:
                    label = self.idx2label[np.argmax(
                        pred_sequence[1][i][j][1:]
                    ) + 1]
                tmp.append(label)
            label_seq.append(tmp)
        return label_seq

    def get_greedy_label(self, pred_sequence, input_data):
        """
        Get label sequence in greedy fashion
        :param pred_sequence: 3D numpy array, prediction results
        :param input_data: list of list of string, tokenized input corpus
        :return label_seq: list of list of string, readable label sequence
        """
        label_seq = []
        for i, s in enumerate(pred_sequence):
            tmp_pred = []
            for j, w in enumerate(s):
                if j < len(input_data[i]):
                    tmp_pred.append(self.idx2label[np.argmax(w[1:])+1])
            label_seq.append(tmp_pred)
        return label_seq

    def devectorize_label(self, pred_sequence, input_data):
        """
        Get readable label sequence
        :param pred_sequence: 4 length list, prediction results from CRF layer
        :param input_data: list of list of string, tokenized input corpus
        :return label_seq: list of list of string, readable label sequence
        """
        if self.crf:
            return self.get_crf_label(pred_sequence, input_data)
        else:
            return self.get_greedy_label(pred_sequence, input_data)

    def prepare_data(self, X, y=None):
        """
        Prepare input and label data for the input
        :param X: list of list of string, tokenized input corpus
        :param y: list of list of string, label sequence
        :return X_input: dict, data input
        :return y_vector: numpy array/None, vector of label data
        """
        vector_word_X, vector_char_X = self.vectorize_input(X)
        X_input = {"word": vector_word_X}
        if self.char_embedding:
            X_input["char"] = vector_char_X

        vector_y = None
        if y is not None:
            vector_y = self.vectorize_label(y)
        return X_input, vector_y

    def train(self, X, y, n_epoch=10, valid_split=None, batch_size=128):
        """
        Prepare input and label data for the input
        :param X: list of list of string, tokenized input corpus
        :param y: list of list of string, label sequence
        :param n_epoch: int, number of training epoch
        :param valid_split: tuple, validation data
            shape: (X_validation, y_validation)
        :param batch_size: int, size of the batch
        :return history: output of the fit method
        """
        self.__init_l2i(y)
        if self.char_embedding:
            self.__init_c2i()

        self.__init_w2i(X)
        self.__init_embedding()
        self.__init_model(y)
        X_train, y_train = self.prepare_data(X, y)
        if valid_split:
            valid_split = self.prepare_data(valid_split[0], valid_split[1])
            es = EarlyStopping(
                monitor="val_crf_loss" if self.crf else "val_loss",
                patience=10,
                verbose=1,
                mode="min",
                restore_best_weights=True
            )
            callback = [es]
        else:
            callback = []
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=n_epoch,
            validation_data=valid_split,
            verbose=1,
            callbacks=callback
        )
        return history

    def predict(self, X):
        """
        Perform prediction
        :param X: list of list of string, tokenized input data
        :return label_result: list of list of string, prediction results
        """
        X_test, _ = self.prepare_data(X)
        pred_result = self.model.predict(X_test)
        label_result = self.devectorize_label(pred_result, X)
        return label_result

    def save(self, filepath):
        """
        Write the model and class parameter into a zip file
        :param filepath: string, path of saved file with ".zip" format
        """
        filename = filepath.split("/")[-1].split(".")[0]
        filenames = {
            "model": f"{filename}_network",
            "class_param": f"{filename}_class.pkl"
        }
        with TemporaryDirectory() as tmp_dir:
            class_param = {
                "label2idx": self.label2idx,
                "word2idx": self.word2idx,
                "seq_length": self.seq_length,
                "word_length": self.word_length,
                "idx2label": self.idx2label,
                "crf": self.crf,
                "char_embedding": self.char_embedding
            }
            if self.char_embedding:
                class_param["char2idx"] = self.char2idx
            with open(f"{tmp_dir}/{filenames['class_param']}", "wb") as pkl:
                pickle.dump(class_param, pkl)
            network_path = f"{tmp_dir}/{filenames['model']}"
            self.model.save(network_path, save_format="tf")
            with ZipFile(filepath, "w") as zipf:
                zipf.write(
                    f"{tmp_dir}/{filenames['class_param']}",
                    filenames['class_param']
                )
                for dirpath, dirs, files in os.walk(network_path):
                    if files == []:
                        zipf.write(
                            dirpath, "/".join(dirpath.split("/")[-2:])+"/"
                        )
                    for f in files:
                        fn = os.path.join(dirpath, f)
                        zipf.write(fn, "/".join(fn.split("/")[3:]))

    @staticmethod
    def load(filepath):
        """
        Load model from the saved zipfile
        :param filepath: path to model zip file
        :return classifier: Loaded model class
        """
        with ZipFile(filepath, "r") as zipf:
            filelist = zipf.filelist
            model_dir = ""
            with TemporaryDirectory() as tmp_dir:
                for fn in filelist:
                    filename = fn.filename
                    if filename.endswith("_class.pkl"):
                        with zipf.open(filename, "r") as pkl:
                            pickle_content = pkl.read()
                            class_param = pickle.loads(pickle_content)
                    elif filename.split("/")[0].endswith("_network"):
                        model_dir = filename.split("/")[0]
                        zipf.extract(filename, tmp_dir)
                model = load_model(
                    f"{tmp_dir}/{model_dir}",
                    custom_objects={
                        "ModelWithCRFLoss": ModelWithCRFLoss
                    }
                )
                model.summary()
            constructor_param = {
                "seq_length": class_param["seq_length"],
                "word_length": class_param["word_length"],
                "crf": class_param["crf"],
                "char_embedding": class_param["char_embedding"]
            }
            classifier = DLHybridTagger(**constructor_param)
            classifier.model = model
            classifier.label2idx = class_param["label2idx"]
            classifier.word2idx = class_param["word2idx"]
            classifier.idx2label = class_param["idx2label"]
            classifier.n_label = len(classifier.label2idx)
            if "char2idx" in class_param:
                classifier.char2idx = class_param["char2idx"]
        return classifier
