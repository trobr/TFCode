import tensorflow as tf

from numpy.random import RandomState

bitch_size = 8
step = 100001

w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

x = tf.placeholder(tf.float32, shape = (None, 2), name = "x-input")
b_ = tf.placeholder(tf.float32, shape = (None, 1), name = "b-input")

a = tf.matmul(x, w1)
b = tf.matmul(a, w2)

cross_entropy = -tf.reduce_mean(b_ * tf.log(tf.clip_by_value(b, 1e-10, 1.0)))
learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 3000
X = rdm.rand(dataset_size, 2)
B = [[int(x1 + x2 < 1)] for (x1, x2) in X]


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print(sess.run(w1))
	print(sess.run(w2))
	print(len(B))
	for i in range(step):
		start = (i * bitch_size) % dataset_size
		end = min(start + bitch_size, dataset_size)

		#print("i : %d   start : %d   end : %d" %(i, start, end))
		sess.run(train_step, feed_dict = {x : X[start : end], b_ : B[start : end]})

		if i % 1000 == 0:
			total_cross_entropy = sess.run(cross_entropy, feed_dict = {x : X, b_ : B})
			print("After %d train, cross_entropy is %g" %(i, total_cross_entropy))

#import tensorflow as tf
# 
#w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
#w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))
#x = tf.placeholder(tf.float32, shape = (1, 2), name = 'input')
#
#a = tf.matmul(x, w1)
#b = tf.matmul(a, w2)
#
#with tf.Session() as sess:
#	init_op = tf.global_variables_initializer()
#	sess.run(init_op)
#	print(sess.run(b, feed_dict = {x : [[0.7, 0.9]]}))