from tensorflow.examples.tutorials.mnist import input_data
# 加载数据
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# 准备变量
import tensorflow as tf
sess = tf.InteractiveSession() #创建新的InteractiveSession == session
x = tf.placeholder(tf.float32,[None,784]) # 第一个参数数据类型，第二个，None不限制行，有784列
# 对W有点不理解
W = tf.Variable(tf.zeros([784,10])) #Variable,存储模型参数，不同与tensor,tensor->一旦使用就会消失，Variable是持久化的
                                    #784：特征维度，10：类别
b = tf.Variable(tf.zeros([10]))     #10：one-hot,10行没有列属性



# 实现算法
y = tf.nn.softmax(tf.matmul(x,W) + b)# matmul：矩阵相乘 P50也公式
 #定义cross-entropy 损失函数
y_ = tf.placeholder(tf.float32,[None,10]) # 声明变量
cross_entrupy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))#p51有详细解释
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entrupy) #设置学习速率0.5
tf.global_variables_initializer().run() # 使用全局优化器，并使用run

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)# 每次选100个数据
    train_step.run({x: batch_xs, y: batch_ys}) # 训练算法

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # 判断条件
accuracy = tf.reduce_mean(tf.case(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))


