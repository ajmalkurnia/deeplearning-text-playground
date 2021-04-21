from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

from .base_tagger import BaseTagger


class BaseCRFTagger(BaseTagger):
    def __init__(self, **kwargs):
        super(BaseCRFTagger, self).__init__(**kwargs)

    def compute_transition_matrix(self, y):
        """
        Initialized transition matrix for CRF

        :param y: 2D list, label of the dataset
        :return transition_matrix: numpy array [n_label+1, n_label+1],
            Transition matrix of the training label
        """
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
        return out_seq

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
        :param pred_sequence: np.array/list, prediction results from out layer
            np.array when using softmax
            list when using CRF
        :param input_data: list of list of string, tokenized input corpus
        :return label_seq: list of list of string, readable label sequence
        """
        return self.get_crf_label(pred_sequence, input_data)

    def training_prep(self, X, y):
        """
        Initialized necessary class attributes for training

        :param X: 2D list, training dataset in form of tokenized corpus
        :param y: 2D list, training data label
        """
        self.init_inverse_indexes(X, y)
        self.embedding, self.embedding_size = self.init_embedding(
            self.embedding_file, self.embedding_type
        )
        self.compute_transition_matrix(y)
        self.init_model()

    def save_crf(self, filepath, zipf):
        """
        Saving CRF model

        :param filepath: string, zip file path
        :param zipf: zipfile
        """
        for dirpath, dirs, files in os.walk(filepath):
            if files == []:
                zipf.write(
                    dirpath, "/".join(dirpath.split("/")[-2:])+"/"
                )
            for f in files:
                fn = os.path.join(dirpath, f)
                zipf.write(fn, "/".join(fn.split("/")[3:]))

    def save_network(self, filepath, zipf):
        """
        Saving keras model

        :param filepath: string, save file path
        :param zipf: zipfile
        """
        self.model.save(filepath[:-5], save_format="tf")
        self.save_crf(filepath[:-5], zipf)
