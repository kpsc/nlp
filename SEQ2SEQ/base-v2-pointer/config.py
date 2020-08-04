# encoding: utf-8

n_epoch = 100
batch_size = 64
keep_prob = 0.9

data_config = {
    "src_len": 30,
    "tgt_len": 30,
    "vocab": "../data/vocab_pg",
    "path": "../data/dialog_pg.train",
}

model_config = {
    "hidden_size": 128,
    "cell_type": "lstm",
    "num_layers": 1,
    "embedding_size": 128,
    "share_embedding": True,
    "max_decode_step": 30,
    "max_grad_norm": 3.0,
    "learning_rate": 0.001,
    "decay_step": 10000,
    "min_learning_rate": 1e-5,
    "bidirection": True,
    "beam_size": 4,
    'keep_prob': 0.9,
    'batch_size': 64
}
