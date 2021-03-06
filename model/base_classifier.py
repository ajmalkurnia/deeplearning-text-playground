from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import Constant

from zipfile import ZipFile
from tempfile import TemporaryDirectory
import numpy as np
import pickle
import itertools
import logging

from common.tokenization import Tokenizer
from common.word_vector import WE_TYPE
from model.AttentionText.attention_text import Attention
from model.TransformerText.transformer_block import (
    TransformerBlock, TransformerEmbedding, MultiAttention
)


class BaseClassifier():
    def __init__(
        self, input_size=50, optimizer="adam", loss="categorical_crossentropy",
        embedding_matrix=None, vocab_size=0, vocab=None, embedding_file=None,
        embedding_type="glorot_uniform", embedding_size=100,
        train_embedding=True
    ):
        """
        Class constructor

            :param input_size: int, maximum number of token input
            :param optimizer: string, learning optimizer of the model
                ("optimizer" parameter during compile)
            :param loss: string, loss function of the model
                ("loss" parameter during compile)
            :param embedding matrix: numpy array,
                Custom embedding matrix of the provided vocab
            :param vocab size: int, maximum size of vocabulary of the model
                (most frequent word of the training data will be used)
            :param vocab: dictionary, the vocabulary index, a dictionary with
                token as keys, and index as value
            :param embedding_file: string, path to embedding file
            :param embedding_type: string, embedding type
                w2v for word2vec, matrix will be taken from embedding file
                ft for FasText, matrix will be taken from embedding file
                onehot, initialize one hot encoding of vocabulary
                custom, use embedding matrix
                or any valid keras.initializer string
            :param emebdding_size: int, the size of embedding output
                only useful when using non-pretrained/generated embedding
            :param train_embedding: boolean,
                trainable parameter on Embedding layer
                which apparently not recommended when using pretrained weight
                refer->https://keras.io/examples/nlp/pretrained_word_embeddings/
        """
        self.vocab_size = vocab_size
        self.label2idx = None
        self.model = None
        self.max_input = input_size
        self.optimizer = optimizer
        self.loss = loss
        # TODO: Test predefined embedding+vocab
        # It should work but not tested yet
        self.embedding = embedding_matrix
        self.vocab = vocab
        self.train_embedding = train_embedding
        self.embedding_type = embedding_type
        self.embedding_size = embedding_size
        if self.embedding:
            self.embedding_size = self.embedding.shape[1]
            self.vocab_size = len(vocab)
            if self.vocab is None:
                raise ValueError(
                    "Provide the vocab if embedding_matrix is not None"
                )
        if embedding_file:
            self.embedding_file = embedding_file
            if self.embedding_type is None:
                raise ValueError(
                    "Provide the embedding_type of the embedding file [w2v|ft]"
                )
        self.logger = logging.getLogger(__name__)

    def init_model(self):
        raise NotImplementedError()

    def init_onehot_embedding(self,):
        """
        Initialization one hot vector the vocabulary
        """
        embedding = np.eye(
            self.vocab_size, dtype=np.int32
        )
        return embedding, len(self.vocab)

    def init_wv_embedding(self, embedding_file, embedding_type):
        """
        Initialization of for Word embedding matrix
        UNK word will be initialized randomly
        """
        wv_model = WE_TYPE[embedding_type].load_model(embedding_file)
        embedding = np.zeros((self.vocab_size, wv_model.size), dtype=float)

        for word, idx in self.vocab.items():
            embedding[idx, :] = wv_model.retrieve_vector(word)
        return embedding, wv_model.size

    def __init_embedding(self):
        if self.embedding_type in ["w2v", "ft", "glove"]:
            self.embedding, self.embedding_size = self.init_wv_embedding(
                self.embedding_file, self.embedding_type
            )
            self.embedding = Constant(self.embedding)
        elif self.embedding_type == "onehot":
            self.embedding, self.embedding_size = self.init_onehot_embedding()
            self.embedding = Constant(self.embedding)
        elif self.embedding_type == "custom":
            self.embedding = Constant(self.embedding)
        else:
            self.embedding = self.embedding_type

    def __init_w2i(self, tokenized_corpus):
        """
        Initialization of vocabulary and the index of the vocabulary
        :param tokenized_corpus: string inside 2 depth list or 3 depth list.
            Ex 2d: [["token", "seq", "data1"]]
            Ex 3d: [[
                ["token", "seq", "sent1", "data1"],
                ["token", "seq", "sent2", "data1]
            ]]

        """
        # Handle 3D tokenized input
        if isinstance(tokenized_corpus[0][0], list):
            tokenized_corpus = [
                list(itertools.chain.from_iterable(data))
                for data in tokenized_corpus
            ]
        special_token = self.add_special_token()
        tokenizer = Tokenizer(self.vocab_size)
        for idx, token in enumerate(special_token):
            tokenizer.vocab_index[token] = idx + 1
        tokenizer.build_vocab(tokenized_corpus)
        self.vocab = tokenizer.vocab_index
        if len(self.vocab) < self.vocab_size:
            self.vocab_size = len(self.vocab)

    def add_special_token(self):
        return ["[UNK]"]

    def vectorized_input(self, corpus):
        """
        Convert sequence of token to sequence of indexes using the vocab_index
        :param corpus: list of list of string, list of tokenized data
        :return idx_list: list of integer
        """
        v_input = []
        for text in corpus:
            idx_list = []
            for idx, token in enumerate(text):
                idx_list.append(self.vocab.get(token, 1))
            v_input.append(idx_list)
        return pad_sequences(v_input, self.max_input, padding='post')

    def __init_l2i(self, target_data):
        """
        Create human readable label index and vice-versa
        :param target_data: list of string
        """
        label = sorted([*{*target_data}])
        self.n_label = len(label)
        self.label2idx = {ch: idx for idx, ch in enumerate(label)}
        self.idx2label = {idx: ch for ch, idx in self.label2idx.items()}

    def vectorized_label(self, label_data):
        """
        Convert human readable label to indexes
        :param label_data: list of string, target label of the dataset
        :return list of int: indexes of the label
        """
        return to_categorical(
            [self.label2idx[label] for label in label_data], self.n_label
        )

    def get_label(self, pred_data):
        """
        Obtain human readable label from prediction results
        :param pred_data: numpy array 2D, prediction results for each output
        return list of string: human readable label of clasification results
        """
        return [self.idx2label[np.argmax(lb)] for lb in pred_data]

    def __prepare_data(self, X, y=None):
        """
        Get the indexes of training data and its label
        :param X: list of list of string, tokenized data
        :param y: list of string, target label of each data
        :return X_feature: list of list of int, sequence of index of the data
        :return y_vector: list of int, indexes of the label
        """
        X_feature = self.vectorized_input(X)
        X_feature = np.array(X_feature, dtype="float32", copy=False)
        y_vector = None
        if y is not None:
            y_vector = self.vectorized_label(y)
        return X_feature, y_vector

    def __train(
        self, X, y, epoch, batch_size,
        validation_pair=None, ckpoint_file=None
    ):
        """
        :param X: list of list of string, tokenized training data
        :param y: list of string, target labels for each training data
        :param epoch: int, number of epoch during training
        :param batch_size: int, the size of batch during training
        :param validation_pair: tupple (val_X, and val_y), validation split
        :param ckpoint_file: string, path to checkpoint
        """
        X, y = self.__prepare_data(X, y)
        if validation_pair:
            val_data = self.__prepare_data(
                validation_pair[0], validation_pair[1]
            )
            es = EarlyStopping(
                monitor='val_accuracy', mode='max', verbose=1,
                patience=int(epoch/5), restore_best_weights=True
            )
            callbacks_list = [es]
            if ckpoint_file:
                checkpoint = ModelCheckpoint(
                    ckpoint_file, monitor='val_accuracy', verbose=1,
                    save_best_only=True, mode='max', period=2
                )
                callbacks_list.append(checkpoint)
        else:
            val_data = None
            callbacks_list = []

        self.model.fit(
            X, y, validation_data=val_data,
            batch_size=batch_size,
            epochs=epoch,
            callbacks=callbacks_list
        )

    def train(
        self, X, y, epoch, batch_size,
        validation_pair=None, ckpoint_file=None
    ):
        """
        Preparing training process
        :param X: list of list of string, tokenized training data
        :param y: list of string, target labels for each training data
        :param epoch: int, number of epoch during training
        :param batch_size: int, the size of batch during training
        :param validation_pair: tupple (val_X, and val_y), validation split
        :param ckpoint_file: string, path to checkpoint
        """
        self.logger.info("Initialized Vocabulary")
        if self.vocab is None:
            self.__init_w2i(X)
        self.logger.info("Initialized embeddings")
        self.__init_embedding()
        self.logger.info("Initialized Label vectors")
        if self.label2idx is None:
            self.__init_l2i(y)
        self.logger.info("Initialized model")
        if self.model is None:
            self.init_model()
        self.logger.info("Run training process")
        self.__train(X, y, epoch, batch_size, validation_pair, ckpoint_file)

    def test(self, X):
        """
        Prediction
        :param X: list of list of stirng, tokenized test data
        :return result: human readable label of classification results
        """
        result = []
        X_test, _ = self.__prepare_data(X)
        self.logger.info("Predicting...")
        y_pred = self.model.predict(X_test)
        result = self.get_label(y_pred)
        return result

    def save(self, filepath):
        """
        Write the model and class parameter into a zip file
        :param filepath: string, path of saved file with ".zip" format
        """
        filename = filepath.split("/")[-1].split(".")[0]
        filenames = {
            "model": f"{filename}.hdf5",
            "class_param": f"{filename}_class.pkl"
        }
        with TemporaryDirectory() as tmp_dir:
            network_path = f"{tmp_dir}/{filenames['model']}"
            self.model.save(network_path)
            class_param = self.get_class_param()
            with open(f"{tmp_dir}/{filenames['class_param']}", "wb") as pkl:
                pickle.dump(class_param, pkl)
            with ZipFile(filepath, "w") as zipf:
                for _, v in filenames.items():
                    zipf.write(f"{tmp_dir}/{v}")

    def get_class_param(self):
        raise NotImplementedError()

    @classmethod
    def load(cls, filepath):
        """
        Load model from the saved zipfile
        :param filepath: path to model zip file
        """
        with ZipFile(filepath, "r") as zipf:
            filelist = zipf.filelist
            for fn in filelist:
                filename = fn.filename
                if filename.endswith(".hdf5"):
                    with TemporaryDirectory() as tmp_dir:
                        zipf.extract(filename, tmp_dir)
                        model = load_model(
                            f"{tmp_dir}/{filename}",
                            custom_objects={
                                "Attention": Attention,
                                "TransformerBlock": TransformerBlock,
                                "TransformerEmbedding": TransformerEmbedding,
                                "MultiAttention": MultiAttention
                                }
                        )
                        model.summary()
                elif filename.endswith("_class.pkl"):
                    with zipf.open(filename, "r") as pkl:
                        pickle_content = pkl.read()
                        class_param = pickle.loads(pickle_content)
            constructor_param = cls.get_construtor_param(class_param)
            classifier = cls(**constructor_param)
            classifier.model = model
            classifier.label2idx = class_param["l2i"]
            classifier.idx2label = class_param["i2l"]
            classifier.n_label = len(classifier.label2idx)
        return classifier

    @staticmethod
    def get_construtor_param(param):
        raise NotImplementedError()
