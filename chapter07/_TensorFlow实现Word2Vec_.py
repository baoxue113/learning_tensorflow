# 采用TSNE进行降维
import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf
import matplotlib.pyplot as plt

url = 'http://mattmahoney.net/dc/'

# 下载数据，我直接用浏览器下载了
# def maybe_download(filename,expected_bytes):
#     if not os.path.exists(filename):
#         filename, _= urllib.request.urlretrieve(url + filename,filename)
#     statinfo = os.stat(filename)
#     if statinfo.st_size == expected_bytes:
#         print('Found and verified',filename)
#     else:
#         print(statinfo.st_size)
#         raise Exception('Failed to verify' + filename + '. Can you get to it with a browser?')
#     return filename
#
# filename = maybe_download('text8.zip',31344016)

# 读取数据
def read_data():
    with zipfile.ZipFile('/Users/zhangfan/kaifa/pycharm/learning_tensorflow/chapter07/text8.zip','r') as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data()
print('Data size',len(words)) # 有多少个单词

vocabulary_size = 50000
def build_dataset(words): # 建造数据集
    count = [['UNK',-1]] #未在字典中的单词个数
    temp1 = collections.Counter(words) # 计数器http://www.2cto.com/kf/201303/196938.html
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))# most_common类似于 top K ,http://www.pythoner.com/205.html
    dictionary = dict() # 单词字典
    for word, _ in count: # 内容匹配
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0 # 不存在字典中的单词
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count # 改数据中的值
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
data, count, dictionary, reverse_dictionary = build_dataset(words)

del words # 删除单词节约内存
print('Most common words (+UNK)',count[:5])
print('Sample data',data[:10],[reverse_dictionary[i] for i in data[:10]])

data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
    global data_index # http://blog.csdn.net/mldxs/article/details/8559973
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape = (batch_size), dtype = np.int32)# 创建矩阵
    labels = np.ndarray(shape = (batch_size, 1), dtype=np.int32)# 创建矩阵
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips): # //：除法取整
        target = skip_window
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[ i * num_skips + j] = buffer[ skip_window ]
            labels[ i * num_skips + j, 0 ] = buffer[ target ]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size = 8, num_skips = 2,skip_window = 1)# p164
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i,0]])

batch_size = 128
embedding_size = 128 # 将单词转为稠密向量的维度
skip_window = 1 # 单词之间的距离
num_skips = 2 # 对每个单词提取的样本数

valid_size = 16 # 验证的单词数
valid_window = 100 # 取最高的100个单词
valid_examples = np.random.choice(valid_window, valid_size, replace = False)# 随机抽取函数
num_sampled = 64 #负样本噪声单词的数量

graph = tf.Graph()
with graph.as_default():
    # placeholder：占位符
    train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
    train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])
    # constant：常量
    valid_dataset = tf.constant(valid_examples, dtype = tf.int32)

    # 用cpu计算
    with tf.device('/cpu:0'): # http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/using_gpu.html,http://blog.csdn.net/queenazh/article/details/52894646
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Variable：变量
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev = 1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # 重要步骤
    loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights, biases = nce_biases,labels = train_labels,inputs = embed,num_sampled = num_sampled, num_classes = vocabulary_size))# 训练的优化目标

    # 随机梯度下降算法SGD
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)#loss损失函数

    # sqrt：开平方根
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims = True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings,transpose_b = True)

    init = tf.global_variables_initializer()

    num_steps = 100001 # 迭代的数次
    with tf.Session(graph = graph) as session:
        init.run()
        print("Initialized")

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
        # 计算loss_val的值
        _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)# 训练算法
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i,:]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
        final_embeddings = normalized_embeddings.eval() #,http://blog.csdn.net/xierhacker/article/details/53103979

def plot_with_labels(low_dim_embs,labels,filename = 'tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize = (18,18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label, xy = (x,y),xytext = (5, 2),textcoords = 'offset points',ha = 'right', va = 'bottom')
    plt.savefig(filename)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# 降维操作
tsne = TSNE(perplexity = 30,n_components = 2, init = 'pca', n_iter = 5000)
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)