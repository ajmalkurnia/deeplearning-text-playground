from keras.utils import to_categorical
from keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.callbacks import EarlyStopping

from common.word_vector import WE_TYPE

from tempfile import TemporaryDirectory
from collections import defaultdict
from zipfile import ZipFile
import numpy as np
import pickle

from .extras.crf_subclass_model import ModelWithCRFLoss


class BaseTagger():
    def __init__(
        self, seq_length=100, embedding_size=100, embedding_file=None,
        embedding_matrix=None, embedding_type="glorot_normal",
        vocabulary=None, vocab_size=10000,
        optimizer="adam", loss="categorical_crossentropy",
    ):
        """
        Deep learning based sequence tagger.
        :param seq_length: int, maximum sequence length in a data
        :param word_embed_size: int, the size of word level embedding,
            relevant when not using pretrained embedding file
        :param embedding_file: string, path to pretrained word embedding
        :param embedding_type: string, word embedding types:
            random, supply any keras initilaizer string
            pretrained, word embedding type of the embedding_file,
                available option: "w2v", "ft", "glove", "onehot", "custom"
        :param vocabulary: dict, inverse index of vocabulary
        :param vocab_size: int, the size of vobulary for the embedding
        :param optimizer: string/object, any valid optimizer parameter
            during model compilation
        :param loss: string/object, any valid loss parameter
            during model compilation
        """

        self.seq_length = seq_length
        # Will be overide with pretrained file embedding
        self.embedding_size = embedding_size
        self.embedding_file = embedding_file
        self.embedding_type = embedding_type

        if embedding_type in ["w2v", "ft", "glove"] and embedding_file is None:
            raise ValueError(
                "Supply parameter embedding_file when using w2v/ft/glove embedding"  # noqa
            )
        self.word2idx = vocabulary
        if vocabulary:
            self.n_words = len(vocabulary)
        if embedding_matrix and vocabulary:
            self.embedding = embedding_matrix
        elif embedding_matrix:
            raise ValueError("Supply the vocab of embedding matrix")

        self.loss = loss

        self.optimizer = optimizer
        self.vocab_size = vocab_size

    def init_w2i(self, data):
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

    def init_l2i(self, data):
        """
        Initialize label to index
        """
        label = list(set([lb for sub in data for lb in sub]))
        self.n_label = len(label)
        self.label2idx = {ch: idx+1 for idx, ch in enumerate(sorted(label))}
        self.idx2label = {idx: ch for ch, idx in self.label2idx.items()}

    def init_wv_embedding(self):
        """
        Initialization of for Word embedding matrix
        UNK word will be initialized randomly
        """
        wv_model = WE_TYPE[self.embedding_type].load_model(self.embedding_file)
        self.word_embed_size = wv_model.size

        self.embedding = np.zeros(
            (self.vocab_size+1, wv_model.size), dtype=float
        )
        for word, idx in self.word2idx.items():
            self.embedding[idx, :] = wv_model.retrieve_vector(word)

    def init_onehot_embedding(self):
        """
        Initialization one hot vector the vocabulary
        """
        self.embedding_size = len(self.vocab)
        self.embedding = np.eye(
            self.vocab_size, dtype=np.int32
        )

    def init_embedding(self):
        """
        Initialize argument for word embedding initializer
        """
        if self.embedding_type in ["w2v", "ft"]:
            self.init_wv_embedding()
            self.embedding = Constant(self.embedding)
        elif self.embedding_type == "onehot":
            self.init_onehot_embedding()
            self.embedding = Constant(self.embedding)
        elif self.embedding_type == "custom":
            self.embedding = Constant(self.embedding)
        else:
            self.embedding = self.embedding_type

    def get_word_vector(self, inp_seq):
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
        """
        word_vector = self.get_word_vector(inp_seq)
        return word_vector

    def vectorize_label(self, out_seq):
        """
        Get prepare vector of the label for training
        :param out_seq: list of list of string, tokenized input corpus
        :return out_seq: 3D numpy array, vector of label data
        """
        out_seq = [[self.label2idx[w] for w in s] for s in out_seq]
        out_seq = pad_sequences(
            maxlen=self.seq_length, sequences=out_seq, padding="post"
        )
        out_seq = [
            to_categorical(i, num_classes=self.n_label+1) for i in out_seq
        ]
        return np.array(out_seq)

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
        :param pred_sequence: 3D numpy array, prediction results from out layer
        :param input_data: list of list of string, tokenized input corpus
        :return label_seq: list of list of string, readable label sequence
        """
        return self.get_greedy_label(pred_sequence, input_data)

    def prepare_data(self, X, y=None):
        """
        Prepare input and label data for the input
        :param X: list of list of string, tokenized input corpus
        :param y: list of list of string, label sequence
        :return X_input: dict, data input
        :return y_vector: numpy array/None, vector of label data
        """
        X_input = self.vectorize_input(X)

        vector_y = None
        if y is not None:
            vector_y = self.vectorize_label(y)
        return X_input, vector_y

    def init_training(self, X, y):
        """
        Initialized necessary class attributes for training

        :param X: 2D list, training dataset in form of tokenized corpus
        :param y: 2D list, training data label
        """
        self.init_l2i(y)
        if self.word2idx is None:
            self.init_w2i(X)

        self.init_embedding()
        self.init_model()

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
        self.init_training(X, y)

        X_train, y_train = self.prepare_data(X, y)
        if valid_split:
            valid_split = self.prepare_data(valid_split[0], valid_split[1])
            es = EarlyStopping(
                monitor="val_loss",
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

    def save_class(self, filepath, zipf):
        """
        Save necessary class attributes
        :param filepath: string, save file path
        :param zipf: ZipFile
        """
        class_param = self.get_class_param()
        with open(filepath, "wb") as pkl:
            pickle.dump(class_param, pkl)
        zipf.write(filepath, filepath.split("/")[-1])

    def save_network(self, filepath, zipf):
        """
        Save keras model
        :param filepath: string, save file path
        :param zipf: ZipFile
        """
        self.model.save(filepath)
        zipf.write(filepath, filepath.split("/")[-1])

    def save(self, filepath):
        """
        Write the model and class parameter into a zip file
        :param filepath: string, path of saved file with ".zip" format
        """
        filename = filepath.split("/")[-1].split(".")[0]
        filenames = {
            "model": f"{filename}_network.hdf5",
            "class_param": f"{filename}_class.pkl"
        }
        with TemporaryDirectory() as tmp_dir:
            with ZipFile(filepath, "w") as zipf:
                class_path = f"{tmp_dir}/{filenames['class_param']}"
                self.save_class(class_path, zipf)
                network_path = f"{tmp_dir}/{filenames['model']}"
                self.save_network(network_path, zipf)

    @classmethod
    def load(cls, filepath):
        """
        Load model from the saved zipfile
        :param filepath: path to model zip file
        :return classifier: Loaded model class
        """
        with ZipFile(filepath, "r") as zipf:
            filelist = zipf.filelist
            with TemporaryDirectory() as tmp_dir:
                for fn in filelist:
                    filename = fn.filename
                    if filename.endswith("_class.pkl"):
                        with zipf.open(filename, "r") as pkl:
                            pickle_content = pkl.read()
                            class_param = pickle.loads(pickle_content)
                    elif filename.split("/")[0].endswith("_network"):
                        zipf.extract(filename, tmp_dir)
                        model_path = filename.split("/")[0]
                    elif filename.endswith(".hdf5"):
                        zipf.extract(filename, tmp_dir)
                        model_path = filename.split("/")[-1]
                model = load_model(
                    f"{tmp_dir}/{model_path}",
                    custom_objects={
                        "ModelWithCRFLoss": ModelWithCRFLoss
                    }
                )
                model.summary()
            classifier = cls.init_from_config(class_param)
            classifier.model = model
        return classifier
