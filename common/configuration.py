from keras import optimizers as opt

BASE_CONFIG = {
    "loss": "categorical_crossentropy",
    "optimizer": "adam",
    "vocab_size": 10000,
    "input_size": 100,  # will be updated according to the dataset
    "embedding_size": 100,  # Overritten when using pretrained embedding
}


OPTIMIZER_CONFIG = {
    "learning_rate": 0.001,
    "clipvalue": None,
    "momentum": 0.9,  # SGD only
    "beta_1": 0.9,  # Adam & Nadam
    "beta_2": 0.9999,  # Adam & Nadam
    "rho": 0.95,  # Adadelta
    "initial_accumulator_value": 0.1,  # Adagrad
}


ARCHITECTURE_CONFIG = {
    "classification": {
        "cnn": {
            "conv_type": "parallel",
            "conv_layers": [
                (256, 2, 1, "relu"),
                (256, 3, 1, "relu"),
                (256, 4, 1, "relu"),
                (256, 5, 1, "relu"),
                (256, 6, 1, "relu")
            ],
            "fcn_layers": [(512, 0.5, "relu")]
        },
        "rnn": {
            "rnn_size": 100,
            "dropout": 0.5,
            "rnn_type": "lstm",
            "attention": None,
            "fcn_layers": [(128, 0.1, "relu")]
        },
        "han": {
            "rnn_size": 100,
            "dropout": 0.2,
            "rnn_type": "gru",
        },
        "rcnn": {
            "rnn_size": 100,
            "rnn_type": "lstm",
            "conv_filter": 128,
            "fcn_layers": [(256, 0.2, "relu")]
        },
        "transformer": {
            "dropout": 0.5,
            "n_blocks": 2,
            "n_heads": 8,
            "dim_ff": 256,
            "pos_embedding_init": True,
            "fcn_layers": [(128, 0.1, "relu")],
            "sequence_embedding": "global_avg"
        },
    },
    "tagger": {
        "hybrid": {
            "word_length": 50,
            "char_embed_size": 30,
            "char_conv_config": [[30, 3, -1], [30, 2, -1], [30, 4, -1]],
            "char_trans_block": 1,
            "char_trans_head": 3,
            "char_trans_dim_ff": 60,
            "char_trans_dropout": 0.3,
            "char_attention_dropout": 0.5,
            "char_trans_scale": False,
            "char_rnn_units": 25,
            "char_recurrent_dropout": 0.33,
            "trans_blocks": 2,
            "trans_heads": 8,
            "trans_dim_ff": 256,
            "trans_dropout": 0.5,
            "attention_dropout": 0.5,
            "trans_scale": False,
            "trans_attention_dim": 256,
            "recurrent_dropout": 0.5,
            "rnn_units": 100,
            "fcn_layers": [],
            "embedding_dropout": 0.5,
            "main_layer_dropout": 0.5,
        },
        "rnn": {
            "word_length": 50,
            "char_embed_size": 100,
            "char_rnn_units": 100,
            "char_recurrent_dropout": 0.33,
            "recurrent_dropout": 0.33,
            "rnn_units": 100,
            "embedding_dropout": 0.5,
            "main_layer_dropout": 0.5
        },
        "cnn": {
            "embedding_dropout": 0.5,
            "pre_outlayer_dropout": 0.5,
            "conv_layers": [[[128, 3], [128, 5]], [[256, 5]], [[256, 5]]],
            "domain_embedding_file": None,
            "domain_embedding_type": "glorot_uniform",
            "domain_embedding_matrix": None,
            "domain_embedding_size": 100,
        },
        "idcnn": {
            "embedding_dropout": 0.5,
            "block_out_dropout": 0.5,
            "repeat": 1,
            "conv_layers": [(300, 3, 1), (300, 3, 2), (300, 3, 1)],
            "fcn_layers": [(1024, 0.5, "relu")]
        },
        "tener": {
            "word_length": 50,
            "char_embed_size": 30,
            "char_heads": 3,
            "char_dim_ff": 60,
            "n_blocks": 2,
            "dim_ff": 128,
            "n_heads": 6,
            "attention_dim": 256,
            "transformer_dropout": 0.3,
            "attention_dropout": 0.5,
            "embedding_dropout": 0.5,
            "fcn_layers": [(512, 0.3, "relu")],
            "out_transformer_dropout": 0.3,
            "scale": False
        }
    }
}


def get_optimizer(conf):
    if conf == "sgd":
        config = {
            "learning_rate": OPTIMIZER_CONFIG["learning_rate"],
            "clipvalue": OPTIMIZER_CONFIG["clipvalue"],
            "momentum": OPTIMIZER_CONFIG["momentum"]
        }
        optimizer = opt.SGD(**config)
    elif conf == "adam":
        config = {
            "learning_rate": OPTIMIZER_CONFIG["learning_rate"],
            "clipvalue": OPTIMIZER_CONFIG["clipvalue"],
            "beta_1": OPTIMIZER_CONFIG["beta_1"],
            "beta_2": OPTIMIZER_CONFIG["beta_2"]
        }
        optimizer = opt.Adam(**config)
    elif conf == "adadelta":
        config = {
            "learning_rate": OPTIMIZER_CONFIG["learning_rate"],
            "clipvalue": OPTIMIZER_CONFIG["clipvalue"],
            "rho": OPTIMIZER_CONFIG["rho"]
        }
        optimizer = opt.Adadelta(**config)
    elif conf == "adagrad":
        config = {
            "learning_rate": OPTIMIZER_CONFIG["learning_rate"],
            "clipvalue": OPTIMIZER_CONFIG["clipvalue"],
            "initial_accumulator_value": OPTIMIZER_CONFIG /
            ["initial_accumulator_value"]
        }
        optimizer = opt.Adagrad(**config)
    elif conf == "nadam":
        config = {
            "learning_rate": OPTIMIZER_CONFIG["learning_rate"],
            "clipvalue": OPTIMIZER_CONFIG["clipvalue"],
            "beta_1": OPTIMIZER_CONFIG["beta_1"],
            "beta_2": OPTIMIZER_CONFIG["beta_2"]
        }
        optimizer = opt.Nadam(**config)
    else:
        raise ValueError("Invalid optimizer")
    return optimizer


def get_config(data, args):
    config = BASE_CONFIG
    config["optimizer"] = get_optimizer(config["optimizer"])
    config["embedding_file"] = args.embeddingfile
    config["embedding_type"] = args.embeddingtype
    config["input_size"] = data.get_sequence_length()

    if data.TASK == "tagger":
        config["seq_length"] = config.pop("input_size")

    config = {**config, **ARCHITECTURE_CONFIG[data.TASK][args.architecture]}
    if args.architecture == "hybrid":
        config["char_embed_type"] = args.charlayer
        config["main_layer_type"] = args.mainlayer
        config["use_crf"] = args.outlayer == "crf"

    return config
