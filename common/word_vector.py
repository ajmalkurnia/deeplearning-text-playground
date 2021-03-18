import os
from glob import glob
from zipfile import ZipFile, BadZipFile

from gensim.models import Word2Vec, FastText
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np


class WordEmbedding():
    def __init__(
        self, embedding_type="w2v",
        embedding_size=100, ngram=(3, 6),
        window_size=5, architecture="sg"
    ):
        self.embedding_type = embedding_type
        self.window = window_size
        self.size = embedding_size
        self.model = None
        if architecture == "sg":
            self.skip_gram = True
        else:
            self.skip_gram = False
        if ngram is None:
            ngram = (3, 6)
        self.min_gram = ngram[0]
        self.max_gram = ngram[1]

    def train_embedding(
        self, sentences, n_iter=100, workers=1,
        min_count=3, negative_sample=1
    ):
        if self.embedding_type == "w2v":
            train_corpus = sentences
            if self.model is None:
                self.model = Word2Vec(
                    size=self.size, window=self.window,
                    min_count=min_count, negative=negative_sample,
                    workers=workers, sg=int(self.skip_gram))
                self.model.build_vocab(train_corpus)
            # self.model.build_vocab()
            else:
                self.model.build_vocab(train_corpus, update=True)
        elif self.embedding_type == "ft":
            train_corpus = sentences
            if self.model is None:
                self.model = FastText(
                    sg=int(self.skip_gram), size=self.size,
                    window=self.window, min_count=min_count,
                    min_n=self.min_gram, max_n=self.max_gram,
                    workers=workers, negative=negative_sample)
                self.model.build_vocab(train_corpus)
            else:
                self.model.build_vocab(train_corpus, update=True)
        elif self.embedding_type == "glove":
            raise ValueError("GloVe training not supported use official repo")
        else:
            raise ValueError("Invalid Embedding Type")
        train_corpus = sentences
        self.model.train(
            train_corpus, epochs=n_iter, total_examples=self.model.corpus_count
        )

    def retrieve_vector(self, word):
        try:
            return self.model.wv[word]
        except KeyError:
            return np.random.random(self.size)

    def find_similar_word(self, word, n=10):
        try:
            return self.model.most_similar(positive=[word], topn=n)
        except KeyError:
            return []

    def save_model(self, file_name):
        self.model.save("{}.model".format(file_name))
        we_model_files = glob("{}.model*".format(file_name))
        with ZipFile(file_name, "w") as zipf:
            for we_file in we_model_files:
                zipf.write(we_file)
                os.remove(we_file)

    def load_model(self, file_name):
        try:
            with ZipFile(file_name, "r") as zipf:
                zipf.extractall("/tmp/")
                nl = zipf.namelist()
            fn = [name for name in nl if name.endswith(".model")][0]
            path = "/tmp/" + fn
        except BadZipFile:
            path = file_name

        if self.embedding_type == "w2v":
            self.model = KeyedVectors.load_word2vec_format(path)
        elif self.embedding_type == "ft":
            self.model = FastText.load_fasttext_format(path)
        elif self.embedding_type == "glove":
            """path name: .txt file"""
            try:
                glove_file = datapath(os.path.abspath(path))
                tmp_file = get_tmpfile("/tmp/g2w2v.txt")
                glove2word2vec(glove_file, tmp_file)
                self.model = KeyedVectors.load_word2vec_format(tmp_file)
            except UnicodeDecodeError:
                self.model = KeyedVectors.load(os.path.abspath(path))
        self.size = self.model.wv.vector_size

    def remove_from_vocab(self, word_list):
        new_vectors = []
        new_vocab = {}
        new_index2entity = []
        new_vectors_norm = []
        if self.embedding_type == "ft":
            self.model.wv.init_sims()
            for i in range(len(self.model.wv.vocab)):
                word = self.model.wv.index2entity[i]
                vec = self.model.wv.vectors[i]
                vocab = self.model.wv.vocab[word]
                vec_norm = self.model.wv.vectors_norm[i]
                if word not in word_list:
                    vocab.index = len(new_index2entity)
                    new_index2entity.append(word)
                    new_vocab[word] = vocab
                    new_vectors.append(vec)
                    new_vectors_norm.append(vec_norm)
            self.model.wv.vocab = new_vocab
            self.model.wv.vectors = np.array(new_vectors)
            self.model.wv.index2entity = new_index2entity
            self.model.wv.index2word = new_index2entity
            self.model.wv.vectors_norm = new_vectors_norm
        else:
            self.model.init_sims()
            for i in range(len(self.model.vocab)):
                word = self.model.index2entity[i]
                vec = self.model.vectors[i]
                vocab = self.model.vocab[word]
                vec_norm = self.model.vectors_norm[i]
                if word not in word_list:
                    vocab.index = len(new_index2entity)
                    new_index2entity.append(word)
                    new_vocab[word] = vocab
                    new_vectors.append(vec)
                    new_vectors_norm.append(vec_norm)
            self.model.vocab = new_vocab
            self.model.vectors = np.array(new_vectors)
            self.model.index2entity = new_index2entity
            self.model.index2word = new_index2entity
            self.model.vectors_norm = new_vectors_norm
