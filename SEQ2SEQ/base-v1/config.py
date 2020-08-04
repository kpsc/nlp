# encoding: utf-8

# 训练轮数
n_epoch = 100
# batch样本数
batch_size = 256
# 训练时dropout的保留比例
keep_prob = 0.8

# 有关语料数据的配置
# data_config = {
#     # 问题最短的长度
#     "min_q_len": 1,
#     # 问题最长的长度
#     "max_q_len": 30,
#     # 答案最短的长度
#     "min_a_len": 2,
#     # 答案最长的长度
#     "max_a_len": 30,
#     # 词与索引对应的文件
#     "word2index_path": "data/w2i.pkl",
#     # 原始语料路径
#     "path": "data/data.train",
#     # 原始语料经过预处理之后的保存路径
#     "processed_path": "data/processed_data.pkl",
# }

data_config = {
    "src_len": 30,
    "tgt_len": 30,
    "vocab": "../data/vocab",
    "path": "../data/dialog_s.train",
}

# 有关模型相关参数的配置
model_config = {
    # rnn神经元单元的状态数
    "hidden_size": 128,
    # rnn神经元单元类型，可以为lstm或gru
    "cell_type": "lstm",
    # 编码器和解码器的层数
    "num_layers": 1,
    # 词嵌入的维度
    "embedding_size": 128,
    # 编码器和解码器是否共用词嵌入
    "share_embedding": True,
    # 解码允许的最大步数
    "max_decode_step": 30,
    # 梯度裁剪的阈值
    "max_grad_norm": 3.0,
    # 学习率初始值
    "learning_rate": 0.001,
    "decay_step": 100000,
    # 学习率允许的最小值
    "min_learning_rate": 1e-6,
    # 编码器是否使用双向rnn
    "bidirection": True,
    # BeamSearch时的宽度
    "beam_size": 4,
    'keep_prob': 0.9,
    'batch_size': 64
}