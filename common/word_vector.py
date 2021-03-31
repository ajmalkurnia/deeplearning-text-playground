from gensim.models import KeyedVectors
import fasttext
import numpy as np


class WordEmbedding():
    def __init__(self):
        """
        Constructor method of WordEmbedding class
        This class is only meant to load and read the word vector from
        pretrained file from original word2vec, fasttext, and glove
        implementation
        """
        self.embedding_size = None
        self.model = None
        self.mean = None

    def find_similar_word(self, word, n=10):
        raise NotImplementedError()

    def get_word_vector(self, word, unk="random"):
        """
        Basic implementation to retrieve word vector
        :param word: str, query word
        :param unk: str, how to handle unkown words, available options
            "random": return randomized vector
            "zeros": return zero vector
            "mean": return the mean of the model
            by default it will return None
        :return word_vector: np.array with shape (size, )
        """
        try:
            return self.model[word]
        except KeyError:
            if unk == "random":
                return np.random.normal(size=self.size)
            elif unk == "zeros":
                return np.zeros(shape=self.size)
            elif unk == "mean":
                return self.mean_vector
            else:
                return None

    @staticmethod
    def load_model(path):
        """
        Load .bin formatted pretrained file
        support for other format (.vec, .txt) will be added later

        :param path: str, full path to pretrained file
            available emebdding types
                w2v: Word2Vec
                ft: FastText
                glove: GloVe
        """
        raise NotImplementedError()


class Word2VecWrapper(WordEmbedding):
    def __init__(self):
        super(Word2VecWrapper, self).__init__()

    def find_similar_word(self, word, n=10):
        try:
            return self.model.similar_by_word(word, topn=n)
        except KeyError:
            return []

    @staticmethod
    def load_model(path):
        """
        Load .bin file of the pretrained vector from original word2vec
        implementation
        :param path: str, full path to bin file
        :return we: Word2VecWrapper, instance of the class
        """
        model = KeyedVectors.load_word2vec_format(path, binary=True)
        we = Word2VecWrapper(model)
        we.model = model
        we.size = model.vector_size
        we.mean_vector = np.mean(model.vectors, axis=0)
        return we


class FastTextWrapper(WordEmbedding):
    def __init__(self):
        super(FastTextWrapper, self).__init__()

    def find_similar_word(self, word, n=10):
        try:
            return self.model.get_nearest_neighbor(n)
        except KeyError:
            return []

    @staticmethod
    def load_model(path):
        """
        Load .bin file of the pretrained vector from original fasttext
        implementation
        :param path: str, full path to bin file
        :return we: FastTextWrapper, instance of the class
        """
        model = fasttext.load_model(path)
        we = FastTextWrapper()
        we.model = model
        we.size = model.get_dimension()
        sum_vector = np.zeros(we.size)
        for word in model.words:
            sum_vector += word
        we.mean_vector = sum_vector/len(model.words)
        return we


class GloVeWrapper(WordEmbedding):
    def __init__(self):
        super(GloVeWrapper, self).__init__()

    def find_similar_word(self, word, n=10):
        """
        Find n-most similar words from query based on cosine similarity
        :param word: str, query word
        :param n: int, return top n array
        :return results: list of tuple, list of n closest words
            each tupple consists of:
                word: str,
                sim_score: float
        """
        try:
            vector = self.model[self.inverse_vocab[word]]
            sim = np.dot(self.model, vector)
            sim /= np.linalg.norm(self.model, axis=1)
            sim /= np.linalg.norm(vector)
            return [(self.vocab[i], sim[i]) for i in np.argsort(-sim)[1:n+1]]
        except KeyError:
            return []

    def get_word_vector(self, word, unk="random"):
        """
        Implementation to retrieve word vector
        :param word: str, query word
        :param unk: str, how to handle unkown words, available options
            "random": return randomized vector
            "zeros": return zero vector
            "mean": return the mean of the model
            by default it will return None
        :return word_vector: np.array with shape (size, )
        """
        try:
            return self.model[self.inverse_vocab[word]]
        except KeyError:
            if unk == "random":
                return np.random.normal(size=self.size)
            elif unk == "zeros":
                return np.zeros(shape=self.size)
            elif unk == "mean":
                return self.mean_vector
            else:
                return None

    @staticmethod
    def load_model(path):
        """
        Load .txt file of the pretrained vector from original GloVe
        implementation
        :param path: str, full path to bin file
        :return we: GloVeWrapper, instance of the class
        """
        model = []
        vocab = []
        with open(path, "r") as f:
            for line in f:
                data = line.split(" ")
                if data[0] == "<unk>":
                    continue
                vocab.append(data[0])
                model.append([float(i) for i in data[1:]])
        model = np.array(model, dtype=float, copy=False)
        instance = GloVeWrapper()
        instance.model = model
        instance.size = model.shape[1]
        instance.vocab = vocab
        instance.inverse_vocab = {w: i for i, w in enumerate(vocab)}
        instance.mean = np.mean(model, axis=1)
        return instance
