import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf
import matplotlib.pyplot as plt

# wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
# tar xvf simple-examples.tgz
#
# git clone https://github.com/tensorflow/models.git
# cd models/tutorials/rnn/ptb

import time
import sys
sys.path.append('/Users/zhangfan/kaifa/pycharm/learning_tensorflow/chapter07/models/tutorials/rnn/ptb')
import reader


class PTBInput(object):

    def __init__(self, config, data, name = None):
        self.batch_size = batch_size = config.batch_size
        # num_steps:LSTM的展开步数
        self.num_steps = num_steps = config.num_steps
        # 计算每个epoch内需要多少轮训练的迭代
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        # 获取特征数据，以及标签数据
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name = name)

class PTBModel(object):
    # is_training：训练标记
    # config：配置参数
    # input_:PTBInput类的实例
    def __init__(self, is_training, config, input_):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        # LSTM的节点数
        size = config.hidden_size
        # 词汇表的大小
        vocab_size = config.vocab_size

        def lstm_cell():
            # 设置默认的LSTM单元
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias = 0.0, state_is_tuple= True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                # 接一个Dropout层
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob = config.keep_prob)

        temp1 = [attn_cell() for _ in range(config.num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple = True)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # 用cpu计算
        with tf.device("/cpu:0"): # gpu中未实现下面的功能
            # P179
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype = tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            # 添加一层dropout
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        state = self._initial_state

        # variable：可变的
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):# num_steps：可展开的步数
                if time_step > 0:
                    temp1 = tf.get_variable_scope().reuse_variables()
                    tf.get_variable_scope().reuse_variables() # 每次循环，状态都会更新
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype = tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype = tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        # P180
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(input_.targets, [-1])], [tf.ones([batch_size * num_steps], dtype = tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable = False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost,tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step = tf.contrib.framework.get_or_create_global_step())
        self._new_lr = tf.placeholder(tf.float32, shape = [], name = "new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict = {self._new_lr: lr_value})

    @property # 将方法变成，类似于java中的get的作用
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

class SmallConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

class MediumConfig(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000

class LargeConfig(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000

class TestConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

def run_epoch(session, model, eval_op = None, verbose = False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state
    }

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / model.input.epoch_size, np.exp(costs / iters), iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)

raw_data = reader.ptb_raw_data('simple-examples/data/')
train_data, valid_data, test_data, _ = raw_data

config = SmallConfig()
eval_config = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
        train_input = PTBInput(config = config, data = train_data, name = "TrainInput")
        with tf.variable_scope("Model", reuse = None, initializer = initializer):
            m = PTBModel(is_training = True, config = config, input_ = train_input)

    with tf.name_scope("Valid"):
        valid_input = PTBInput(config = config, data = valid_data, name = "ValidInput")
        with tf.variable_scope("Model", reuse = True, initializer = initializer):
            mvalid = PTBModel(is_training = False, config = config, input_ = valid_input)

    with tf.name_scope("Test"):
        test_input = PTBInput(config = eval_config, data = valid_data, name = "TestInput")
        with tf.variable_scope("Model", reuse = True, initializer = initializer):
            mtest = PTBModel(is_training = False, config = config, input_ = test_input)

    sv = tf.train.Supervisor()
    with sv.managed_session() as session:
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i + 1 - config.max_max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f " % (i + 1, session.run(m.lr)))

            train_perplexity = run_epoch(session, m ,eval_op = m.train_op, verbose = True)
            print("Epoch: %d Learning rate: %.3f " % (i + 1, train_perplexity))

            valid_perplexity = run_epoch(session, mvalid)
            print("Epoch: %d Learning rate: %.3f " % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, mtest)
        print("Test Prplexity: %.3f " % test_perplexity)