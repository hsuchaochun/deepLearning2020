# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

'''
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
'''
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow as image_summary
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from collections import namedtuple

BATCH_SIZE = 100
ITERATION = 1000
OUTPUT_FILE_DIR = './'
OUTPUT_FILE_NAME = 'logs'
L2_REGULARIZATION = False

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_activations(layer, stimuli):
    units=sess.run(layer, feed_dict = {xs: np.reshape(
        stimuli, [1, 784], order='F'), keep_prob: 1.0})
    plot_nn_filter(units)

def plot_nn_filter(units):
    import math
    filters=units.shape[3]
    plt.figure(1, figsize = (20, 20))
    n_columns=6
    n_rows=math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0, :, :, i], interpolation = 'nearest', cmap = 'gray')
    plt.tight_layout()
    plt.show()
    
# define placeholder for inputs to network
xs = tf.compat.v1.placeholder(tf.float32, [None, 784])/255.   # 28x28
ys = tf.compat.v1.placeholder(tf.float32, [None, 10])
keep_prob = tf.compat.v1.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
with tf.name_scope('conv1'):
    with tf.name_scope('weights'):
        # patch 5x5, in size 1, out size 32
        W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')
        tf.summary.histogram('conv1/weights', W_conv1)
    with tf.name_scope('bias'):
        b_conv1 = bias_variable([32], name='b_conv1')
        tf.summary.histogram('conv1/biases', b_conv1)
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) +
                         b_conv1)  # output size 28x28x32
    tf.summary.histogram('conv1/outputs', h_conv1)
    # output size 14x14x32
    h_pool1 = max_pool_2x2(h_conv1)

## conv2 layer ##
with tf.name_scope('conv2'):
    with tf.name_scope('weights'):
        # patch 5x5, in size 32, out size 64
        W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')
        tf.summary.histogram('conv2/weights', W_conv2)
    with tf.name_scope('bias'):
        b_conv2 = bias_variable([64], name='b_conv2')
        tf.summary.histogram('conv2/biases', b_conv2)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) +
                         b_conv2)  # output size 14x14x64
    tf.summary.histogram('conv2/outputs', h_conv2)
    # output size 7x7x64
    h_pool2 = max_pool_2x2(h_conv2)

## fc1 layer ##
with tf.name_scope('fc1'):
    with tf.name_scope('weights'):
        W_fc1 = weight_variable([7*7*64, 1024], name='W_fc1')
        tf.summary.histogram('fc1/weights', W_fc1)
    with tf.name_scope('bias'):
        b_fc1 = bias_variable([1024], name='b_fc1')
        tf.summary.histogram('fc1/biases', b_fc1)
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    tf.summary.histogram('fc1/outputs', h_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
with tf.name_scope('fc2'):
    with tf.name_scope('weights'):
        W_fc2 = weight_variable([1024, 10], name='W_fc2')
        tf.summary.histogram('fc2/weights', W_fc2)
    with tf.name_scope('bias'):
        b_fc2 = bias_variable([10], name='b_fc2')
        tf.summary.histogram('fc2/biases', b_fc2)
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    tf.summary.histogram('fc2/outputs', prediction)


# the error between prediction and real data
with tf.name_scope('loss'):
    l2_loss = 0.005*tf.nn.l2_loss(W_conv1) + 0.005*tf.nn.l2_loss(W_conv2) + 0.005*tf.nn.l2_loss(W_fc1) + 0.005*tf.nn.l2_loss(W_fc2)
    if L2_REGULARIZATION:
        loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction) + l2_loss,
                                                reduction_indices=[1]))
    else:
        loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
    print('!!!!!!loss', loss)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

images=x_image[:100]
tf.summary.image('100 training data examples', images, max_outputs = 100)

sess=tf.Session()
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter(OUTPUT_FILE_DIR + OUTPUT_FILE_NAME, sess.graph)

# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init=tf.initialize_all_variables()
else:
    init=tf.global_variables_initializer()
sess.run(init)

for i in range(ITERATION):
    batch_xs, batch_ys=mnist.train.next_batch(BATCH_SIZE)
    print(batch_xs)
    print(batch_ys)
    result = sess.run(train_step, feed_dict = {
                    xs: batch_xs, ys: batch_ys, keep_prob: 0.5})

    if i % 50 == 0:
        result = sess.run(merged, feed_dict = {
                        xs: batch_xs, ys: batch_ys, keep_prob: 0.5})

        y_pre = sess.run(prediction, feed_dict = {
                       xs: mnist.test.images[:1000], keep_prob: 1})
        prediction_array = tf.argmax(y_pre, 1).eval(
            session = tf.compat.v1.Session())
        print('predict number: \n', prediction_array)

        index = np.where(mnist.test.labels == 1)
        print('label number: \n', index[1][:1000])

        accuracy = compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000])
        print('accuracy: ', accuracy)

        writer.add_summary(result, i)

    if i == (ITERATION-1):
        y_pre = sess.run(prediction, feed_dict = {
                       xs: mnist.test.images[:1000], keep_prob: 1})
        correct_prediction=tf.equal(
            tf.argmax(y_pre, 1), tf.argmax(mnist.test.labels[:1000], 1))
        sess.run(tf.print(correct_prediction, summarize=1000))

        false_prediction = tf.math.logical_not(correct_prediction)
        false_index = tf.where(false_prediction)
        false_index = tf.reshape(false_index, [-1])
        false_index_array = false_index.eval(session = tf.compat.v1.Session())
        print('false index: \n', false_index_array)

        get_activations(h_conv1, mnist.test.images[0])
        get_activations(h_conv2, mnist.test.images[0])
