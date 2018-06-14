'''
import tensorflow as tf 
from numpy.random import RandomState

data_size = 128
batch_size = 8
loss_more = 1
loss_less = 10
STEP = 5000

x = tf.placeholder(tf.float32, shape = (None, 2), name = 'input-x')
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = 'input-y')

w1 = tf.Variable(tf.random_normal([2, 1], stddev = 1, seed = 1))
y = tf.matmul(x, w1)

loss = tf.reduce_sum(tf.where(tf.greater(y, y_), 
					(y - y_) * loss_more, 
					(y_ - y) * loss_less))

train = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)
X = rdm.rand(data_size, 2)
Y = [[x1 + x2 + rdm.rand() / 10. - 0.05] for (x1, x2) in X]

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	for i in range(STEP):
		start = (i * batch_size) % data_size
		end = min(start + batch_size, data_size)
		sess.run(train, feed_dict = {x : X[start : end], y_ : Y[start : end]})
	print(sess.run(w1))
'''


''' 
# 5层神经网络带L2正则化
import tensorflow as tf

def get_weight(shape, lambda_):
    weight = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(lambda_)(weight))
    return weight


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

batch_size = 8

layer_dimension = [2, 8, 10, 20, 50]
n_layers = len(layer_demension)
cur_layer = x
cur_dimension = layer_dimension[0]

for i in range(1, n_layers):
    next_dimension = layer_dimension[i]
    weight = get_weight([cur_dimension, next_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[next_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    cur_dimension = next_dimension

mse_loss = tf.reduce_sum(tf.square(cur_layer - y_))
regular = tf.get_collection('losses')
loss = mse_loss + regular
'''

# 平均滑动模型
import tensorflow as tf

v1 = tf.Variable(0.)
num_update = tf.Variable(0, trainable=False)

ema = tf.train.ExponentialMovingAverage(0.99, num_update)
maintian_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(v1, 5))
    sess.run(maintian_op)
    print(sess.run([v1, ema.average(v1)]))
