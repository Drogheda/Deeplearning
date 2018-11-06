import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#x并不是一个固定值而是一个占位符，只有在TensorFlow运行时才会被设定真实值。
#x是一个[任意维，784]的矩阵
x = tf.placeholder(tf.float32, [None, 784])

# Initialize variables(初始化变量)
#tf.Variable()用来创建变量
#W是一个[784, 10]的矩阵。有0-9十类图片，每张图片由784个像素表示
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#Create model(创建模型)
#tf.matmul(x, W)两个矩阵相乘
#tf.nn.softmax()进行归一化计算，得到每张图片在每个分类下的概率
#y是一个[任意维，10]的矩阵
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss func(损失函数)————交叉熵
#增加一个占位符来输入真实分布
y_ = tf.placeholder(tf.float32, [None, 10])
#tf.nn.softmax_cross_entropy_with_logits()函数由下面三步组成
#第一步：先对网络的最后一层输出做softmax；
#第二步：softmax的输出向量和实际标签做一个交叉熵；
#第三步：最后求一个平均值
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))


#定义优化器————梯度递减算法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 启用该模型，并初始化所有变量
#init = tf.initialize_all_variables()
#sess = tf.Session()
#sess.run(init)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# train(训练模型)
#每次允许执行1000次训练
#每一次训练都从训练集中随机抽取100条数据
#执行train_step将占位数据替换成从测试图片库中获取的参数
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

#evaluation model(评估模型)
#tf.arg_max()找到张量y第二个向量的最大值，即找到每一个图片对应的最高概率
#correct_prediction是boolean类型，将其转化成float类型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 这里使用的是整个mnist.test的数据
print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))