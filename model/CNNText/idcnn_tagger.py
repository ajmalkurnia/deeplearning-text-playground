from keras.layers import Embedding, Dropout, Conv1D, Dense
from keras.models import Model, Input
from keras.preprocessing.sequence import pad_sequences
from tensorflow_addons.layers.crf import CRF
import numpy as np
import os

from model.extras.crf_subclass_model import ModelWithCRFLoss
from model.base_tagger import BaseTagger


class IDCNNTagger(BaseTagger):
    def __init__(
        self, embedding_dropout=0.5, block_out_dropout=0.5, repeat=1,
        conv_layers=[(300, 3, 1), (300, 3, 2), (300, 3, 1)],
        fcn_layers=[(1024, 0.5, "relu")], **kwargs
    ):
        """
        CNN based sequence tagger.

        :param embedding_dropout: float, dropout rate after embedding layer
        :param pre_outlayer_dropout: float, dropout rate before output layer
        :param conv_layers: 3D list-like, convolution layer settings,
            each list component denotes config for 1 layer,
                each config consist of 2 length tuple/list that denotes:
                    int, number of filter,
                    int, filter size,
            each convolution will be concatenated for 1 layer,
            each layer is connected sequentially
        """
        super(IDCNNTagger, self).__init__(**kwargs)
        self.ed = embedding_dropout
        self.block_out_dropout = block_out_dropout
        self.conv_layers = conv_layers
        self.fcn_layers = fcn_layers
        self.repeat = repeat
        self.loss = "sparse_categorical_crossentropy"

    def init_model(self):
        """
        Initialize the network model
        """
        # Word Embebedding
        input_layer = Input(shape=(self.seq_length,))
        embedding_layer = Embedding(
            self.vocab_size+1, self.word_embed_size,
            input_length=self.seq_length,
            embeddings_initializer=self.embedding,
            mask_zero=True,
        )
        embedding_layer = embedding_layer(input_layer)
        conv_layer = Dropout(self.ed)(embedding_layer)
        for idx in range(self.repeat):
            for filter_num, filter_size, d in self.conv_layers:
                conv_layer = Conv1D(
                    filter_num, filter_size,
                    dilation_rate=d,
                    activation="relu",
                    padding="same"
                )(conv_layer)
            conv_layer = Dropout(self.block_out_dropout)(conv_layer)
        fcn_layer = conv_layer
        for unit, dropout, activation in self.fcn_layers:
            fcn_layer = Dense(
                unit, activation=activation
            )(fcn_layer)
            fcn_layer = Dropout(dropout)(fcn_layer)
        self.model = Dense(self.n_label+1)(fcn_layer)
        crf = CRF(self.n_label+1)
        out = crf(self.model)
        self.model = Model(inputs=input_layer, outputs=out)
        self.model.summary()
        # Subclassing to properly compute crf loss
        self.model = ModelWithCRFLoss(self.model)
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
        return np.array(out_seq)

    def init_training(self, X, y):
        self.init_l2i(y)
        if self.word2idx is None:
            self.init_w2i(X)
        self.init_embedding()
        self.__init_transition_matrix(y)
        self.init_model()

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

    def devectorize_label(self, pred_sequence, input_data):
        """
        Get readable label sequence
        :param pred_sequence: 4 length list, prediction results from CRF layer
        :param input_data: list of list of string, tokenized input corpus
        :return label_seq: list of list of string, readable label sequence
        """
        return self.get_crf_label(pred_sequence, input_data)

    def save_crf(self, filepath, zipf):
        for dirpath, dirs, files in os.walk(filepath):
            if files == []:
                zipf.write(
                    dirpath, "/".join(dirpath.split("/")[-2:])+"/"
                )
            for f in files:
                fn = os.path.join(dirpath, f)
                zipf.write(fn, "/".join(fn.split("/")[3:]))

    def save_network(self, filepath, zipf):
        self.model.save(filepath[:-5], save_format="tf")
        self.save_crf(filepath[:-5], zipf)

    def get_class_param(self, filepath, zipf):
        class_param = {
            "label2idx": self.label2idx,
            "word2idx": self.word2idx,
            "seq_length": self.seq_length,
            "idx2label": self.idx2label,
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
            "seq_length": class_param["seq_length"]
        }
        classifier = IDCNNTagger(**constructor_param)
        classifier.label2idx = class_param["label2idx"]
        classifier.word2idx = class_param["word2idx"]
        classifier.idx2label = class_param["idx2label"]
        classifier.n_label = len(classifier.label2idx)
        return classifier
