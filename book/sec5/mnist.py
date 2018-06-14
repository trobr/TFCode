#
#  ┏┓	┏┓
# ┏┛┻━━━┛┻┓
# ┃	      ┃
# ┃  ━    ┃
# ┃┳┛  ┗┳ ┃
# ┃	      ┃
# ┃	  ┻   ┃
# ┃	      ┃
# ┗━┓   ┏━┛
#	┃   ┃	神兽保佑
#	┃   ┃	代码无BUG!
#	┃   ┗━━━┓
#	┃ 		┣┓
#	┃ 		┏┛
#	┗┓┓┏━┳┓┏┛
#	 ┃┫┫ ┃┫┫
#	 ┗┻┛ ┗┻┛
#


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100

BASE_LEARNING_RATE = 0.1
DECAY_LEARNING_RATE = 0.99
REGULARIZATION_RATE = 0.0001
TRAIN_STEP = 5000
DECAY_MOVING_AVE = 0.99

# 输出层不能是relu激活函数


def net_struct(in_data, ave, weight1, bias1, weight2, bias2):
    if ave == None:
        layer1 = tf.nn.relu(tf.matmul(in_data, weight1) + bias1)
        return (tf.matmul(layer1, weight2) + bias2)
    else:
        layer1 = tf.nn.relu(
            tf.matmul(in_data, ave.average(weight1)) + ave.average(bias1))
        return (tf.matmul(layer1, ave.average(weight2)) + ave.average(bias2))


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y_input')

    weight1 = tf.Variable(tf.truncated_normal(
        [INPUT_NODE, LAYER1_NODE], stddev=0.1))
    bias1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weight2 = tf.Variable(tf.truncated_normal(
        [LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = net_struct(x, None, weight1, bias1, weight2, bias2)

    global_step = tf.Variable(0, trainable=False)
    variable_ave = tf.train.ExponentialMovingAverage(
        DECAY_MOVING_AVE, global_step)
    variable_ave_op = variable_ave.apply(tf.trainable_variables())
    ave_y = net_struct(x, variable_ave, weight1, bias1, weight2, bias2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regular = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regular(weight1) + regular(weight2)
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(
        DECAY_LEARNING_RATE, global_step,
        mnist.train.num_examples / BATCH_SIZE, DECAY_LEARNING_RATE)

    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)
    train_op = tf.group(train_step, variable_ave_op)

    correct_prediction = tf.equal(tf.argmax(ave_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAIN_STEP):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('after %d train, validation acc using average model is %g' %
                      (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
            print('1')

        test_acc = sess.run(accuracy, feed_dict=validate_feed)
        print('after %d train, validation acc using average model is %g' %
              (TRAIN_STEP, test_acc))
        summary_writer = tf.summary.FileWriter(r'./log',
                                               graph_def=sess.graph_def)
        summary_writer.close()


def main(argv=None):
    mnist = input_data.read_data_sets(
        r'D:\ImgPro\DL&ML\TensorFlow\dataset\MNIST', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
