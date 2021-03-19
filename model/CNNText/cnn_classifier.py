from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout, concatenate
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
import keras

from model.base_classifier import BaseClassifier


class CNNClassifier(BaseClassifier):
    def __init__(
        self, conv_layers=[(256, 5, 2, "relu")],
        fcn_layers=[(512, 0.5, "relu")], conv_type="sequence",
        **kwargs
    ):
        """
        Class constructor
        :param conv_layers: list of tupple,
            A list of parameter for convolution layers,
            each tupple for one convolution layer that consist of : [
                (int) Number of filter,
                (int) filter size,
                (int) maxpooling (-1 to skip),
                (string) activation function
            ]
        :param fcn_layers: list of tupple, parameter for Dense layer
            each tupple for one FC layer,
            final layer (softmax) will be automatically added in,
            each tupple consist of: [
                (int) Number of unit,
                (float) dropout (-1 to skip),
                (string) activation function
            ]
        :param conv_type: string, Set how the convolution will be performed
            available options: parallel/sequence
            parallel: each cnn layer from conv_layers will run against
                embedding matrix directly, the result will be concatenated,
                Refer to Yoon Kim 2014
            sequence: cnn layer from conv_layers will stacked sequentially,
                commonly used for character level CNN,
                on word level CNN parallel is recommended
        :param input_size: int, maximum number of token input
        :param optimizer: string, learning optimizer (keras model "optimizer")
        :param loss: string, loss function
        :param embeding matrix: numpy array
        :param vocab size: int, maximum size of vocabulary of the CNN
            (most frequent word of the training data will be used)
        :param vocab: dict, inverse index of vocabulary {"word":1}
        :param embedding_file: string, path to pretrained emebdding file
        :param embedding_type: string type of embedding file,
            w2v for Word2Vec
            ft for FasText
        """
        self.model = None
        self.conv_layers = conv_layers
        self.conv_type = conv_type
        self.fcn_layers = fcn_layers

        super().__init__(**kwargs)

    def init_model(self):
        "Initialization of the CNN Model"
        inputs = Input(shape=(self.max_input, ), name="inp", dtype='int64')
        # Embedding
        embedding_layer = Embedding(
            self.vocab_size,
            self.embedding_size,
            input_length=self.max_input,
            embeddings_initializer=keras.initializers.Constant(self.embedding)
            # trainable=True
        )
        x = embedding_layer(inputs)
        # 1 is stacked the usual way (like image CNN),
        if self.conv_type == "sequence":
            for n_filter, filter_s, pool_size, activation in self.conv_layers:
                x = Conv1D(n_filter, filter_s)(x)
                x = Activation(activation)(x)
                if pool_size != -1:
                    x = MaxPooling1D(pool_size=pool_size)(x)
            x = Flatten()(x)
        # 2 is a multiple single layer CNN -> concat
        elif self.conv_type == "parallel":
            conv_layers = []
            for n_filter, filter_s, _, activation in self.conv_layers:
                conv_layer = Conv1D(n_filter, filter_s)(x)
                conv_layer = Activation(activation)(conv_layer)
                conv_layer = GlobalMaxPooling1D()(conv_layer)
                conv_layers.append(conv_layer)
            x = concatenate(conv_layers, axis=1)

        # Fully connected layers
        for dense_size, dropout, activation in self.fcn_layers:
            x = Dense(dense_size, activation=activation)(x)
            if dropout > 0:
                x = Dropout(dropout)(x)
        # Output Layer
        predictions = Dense(self.n_label, activation='softmax')(x)
        # Build model
        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['accuracy']
        )
        self.model.summary()

    def get_class_param(self):
        return {
            "input_size": self.max_input,
            "l2i": self.label2idx,
            "i2l": self.idx2label,
            "vocab": self.vocab,
            "embedding_size": self.embedding_size,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "conv_layers": self.conv_layers,
            "conv_type": self.conv_type,
            "fcn_layers": self.fcn_layers
        }

    def load_class_param(self, class_param):
        """
        Load model from the saved zipfile
        :param filepath: path to model zip file
        """
        self.max_input = class_param["input_size"]
        self.label2idx = class_param["l2i"]
        self.idx2label = class_param["i2l"]
        self.vocab = class_param["vocab"]
        self.embedding_size = class_param["embedding_size"]
        self.optimizer = class_param["optimizer"]
        self.loss = class_param["loss"]
        self.conv_layers = class_param["conv_layers"]
        self.conv_type = class_param["conv_type"]
        self.fcn_layers = class_param["fcn_layers"]
        self.n_label = len(self.label2idx)