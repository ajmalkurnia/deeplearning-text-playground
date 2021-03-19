import numpy as np
import scipy
import sklearn_crfsuite
import pickle
import os
from collections import defaultdict
from nltk.tokenize import word_tokenize, casual_tokenize
from zipfile import ZipFile
from sklearn import model_selection
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer


class Tokenizer():
    def __init__(self, vocab_size=0, unk_token=True):
        """
        :param vocab_size: int, maximum vocabulary size, 0 means use every word
        :param unk_token: bool, if True any oov token will be preserved and
            printed as "UNK", False any oov will filtered out
            only used if build_vocab function has been used
        """
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab_count = defaultdict(int)
        self.vocab_index = {}

    def tokenize(self):
        raise NotImplementedError

    def build_vocab(self, tokenized_corpus):
        """
        Find n most frequent token in the corpus as vocabulary,
        tokenization after this function called will handle OOV token
        :param tokenized_corpus: list of list of string, training corpus
        """
        self.vocab_count = self._count_token(tokenized_corpus)
        sorted_vocab = sorted(
            self.vocab_count.items(), key=lambda x: x[1], reverse=True
        )
        for word, _ in sorted_vocab:
            if len(self.vocab_index) < self.vocab_size or self.vocab_size == 0:
                self.vocab_index[word] = len(self.vocab_index)

    def to_index(self, tokenized_corpus):
        """
        convert token in tokenized_corpus to index based on trained word_index
        :param tokenized_corpus: list of list of text, tokenized corpus
        :return indexed_corpus: list of list of int
        """
        indexed_corpus = []
        for tokens in tokenized_corpus:
            idx_data = []
            for token in tokens:
                if token in self.vocab_index:
                    idx_data.append(self.vocab_index[token])
                elif self.unk_token is False:
                    idx_data.append(0)
            indexed_corpus.append(idx_data)
        return indexed_corpus

    def handle_unk(self, tokens):
        if len(self.vocab_index) > 1:
            for idx, token in enumerate(tokens):
                if token not in self.vocab_index and self.unk_token:
                    tokens[idx] = "UNK"
        return tokens

    def _count_token(self, tokenized_corpus):
        """
        Count token frequency
        :param tokenized_corpus: list of list of text, tokenized corpus
        :return counter: defautldict {text: int} token frequenncy counter
        """
        counter = defaultdict(int)
        for data in tokenized_corpus:
            for token in data:
                counter[token] += 1
        return counter


class NLTKToknizerWrapper(Tokenizer):
    def __init__(self, formal=True, vocab_size=0, unk_token=True):
        """
        :param formal: bool, if True the tokenization process uses nltk's
            word_tokenize function, else it will use casual_tokenize function
        :param vocab_size: int, maximum vocabulary size, 0 means use every word
        :param unk_token: bool, if True any oov token will be preserved and
            printed as "UNK", False will filtered out oov from tokenization
            result only used if build_vocab function has been used
        """
        super().__init__(vocab_size, unk_token)
        if formal:
            self.tokenize_func = word_tokenize
        else:
            self.tokenize_func = casual_tokenize

    def tokenize(self, text):
        tokens = self.tokenize_func(text)
        tokens = self.handle_unk(tokens)
        return tokens


class CRFtokenizer:
    def __init__(self, window=5, vocab_size=0, unk_token=True, **kwargs):
        """
        :param window: the size of context window,
        :param vocab_size: int, maximum vocabulary size, 0 means use every word
        :param unk_token: bool, if True any oov token will be preserved
            and printed as "UNK", False will filtered out oov from
            tokenization result,
            only used if build_vocab function has been called
        :param kwargs: crfsuite Parameter
        """
        self.window = window
        super().__init__(vocab_size, unk_token)
        self.model = sklearn_crfsuite.CRF(**kwargs)

    def _ext_feature(self, text):
        """
        :param text: string, document input
        :return features: a list of dict, feature for each character in a text
        """
        features = []
        for i, char in enumerate(text):
            # orthographic feature
            feature = {}

            feature['is_alpha'] = char.isalpha()
            feature['is_numeric'] = char.isnumeric()
            feature['other'] = not char.isalpha() and not char.isnumeric()
            feature["caps"] = char.isupper()
            feature['is_single_quote'] = char == "\'"

            # Window Feature
            if self.window:
                temp_text = "«"+text+"»"
                if i-self.window > 0:
                    window_b = i-self.window
                else:
                    window_b = 0

                if i+self.window+1 < len(temp_text):
                    window_e = i+self.window+1
                else:
                    window_e = len(temp_text)
                for idx, window_ch in enumerate(temp_text[window_b:window_e]):
                    feature[f"char[{window_b+idx-i}]"] = window_ch.lower()
            else:
                feature["char[0]"] = char
            features.append(feature)
        return features

    def _ext_label(self, y_seq):
        """
        converts BIO tag string to list of BIO tag
        :param y_seq : label BIO string
        :return list: list of BIO tag char
        """
        return list(y_seq)

    def train(self, X, y):
        """
        :param X: list of string documents input
        :param y: list of string BIO labels reference output
        """
        X_train = []
        y_train = []
        for doc, label_seq in zip(X, y):
            X_train.append(self._ext_feature(doc))
            y_train.append(self._ext_label(label_seq))
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predict the BIO tag of a list of string
        :param X: list of string, list of text input
        :return y_pred: list of list of char, BIO tag sequence
        """
        X_test = []
        for doc in X:
            X_test.append(self._ext_feature(doc))
        y_pred = self.model.predict(X_test)
        return y_pred

    def tokenize(self, text):
        """
        Predict the BIO tag and decode the BIO tag to produce token
        :param text: string, text input to be tokenized
        :return tokens: list of string, tokenization results for each data
        """
        bio_tags = self.predict([text])
        current_token = ""
        tokens = []
        for char, tag in zip(text, bio_tags):
            if tag == "B" and current_token != "":
                tokens.append(current_token)
                current_token = char
            elif tag != "O":
                current_token += char
        if current_token != "":
            tokens.append(current_token)
        tokens = self.handle_unk(tokens)
        return tokens

    def tuning(self, X, y):
        """
        Hyper paramter tuning, this fuction will tune:
            the regularization factor (c1 and c2)
            maximum iteration
        :param X: list of string documents input
        :param y: list of string BIO labels reference output
        :return tupple: parameter settings max iteration, c1, c2
        """
        X_train = []
        for tweet in X:
            X_train.append(self._ext_feature(tweet))
        grid = {
            "c1": scipy.stats.expon(scale=0.5).rvs(size=10),
            "c2": scipy.stats.expon(scale=0.05).rvs(size=10),
            "max_iterations": np.arange(100, 501, 100),
        }
        labels = list(set([tag for row in y for tag in row]))

        scorer = make_scorer(
            metrics.flat_f1_score,
            average='macro',
            labels=labels
        )
        gs = model_selection.GridSearchCV(
            self.model,
            grid,
            cv=4,
            n_jobs=-1,
            verbose=1,
            scoring=scorer
        )
        gs.fit(X_train, y)
        print("Best Param {}".format(gs.best_params_))
        print("Best CV Model {}".format(gs.best_score_))
        print("Model Size {:0.2f}M".format(gs.best_estimator_.size_ / 1000000))
        max_iterations = gs.best_params_["max_iterations"]
        c1 = gs.best_params_["c1"]
        c2 = gs.best_params_["c2"]

        self.model = sklearn_crfsuite.CRF(
            max_iterations=max_iterations,
            c1=c1,
            c2=c2,
        )
        return max_iterations, c1, c2

    def save_model(self, filename="model/crf_tokenizer.zip"):
        """
        Save CRF model and class paramter to pickle and zip them both together
        :param filename: string, full directory of target file ending with .zip
        """
        file_only = os.path.splitext(os.path.basename(filename))[0]
        tmp_filenames = {
            "crf_model": "/tmp/"+file_only+"_crf.pkl",
            "class_param": "/tmp/"+file_only+"_class.pkl"
        }
        class_param = {
            "verbose_ortho": self.verbose_ortho,
            "window": self.window
        }

        with open(tmp_filenames["crf_model"], "wb") as pkl:
            pickle.dump(self.model, pkl)
        with open(tmp_filenames["class_param"], "wb") as pkl:
            pickle.dump(class_param, pkl)

        zip_file = ZipFile(filename, "w")
        for k, v in tmp_filenames.items():
            zip_file.write(v)

    def load_model(self, filename):
        """
        Load CRF configuration and class parameter from saved pretrained file
        :param filename: string, full path of pretrained model
        """

        with ZipFile(filename, "r") as zipf:
            filelist = zipf.namelist()
            zipf.extractall("/tmp/")
            for fn in filelist:
                if fn.endswith("_crf.pkl"):
                    with open("/tmp/"+fn, "rb") as pkl:
                        self.model = pickle.load(pkl)
                elif fn.endswith("_class.pkl"):
                    with open("/tmp/"+fn, "rb") as pkl:
                        class_param = pickle.load(pkl)
                        self.window = class_param["window"]
                        self.verbose_ortho = class_param["verbose_ortho"]
