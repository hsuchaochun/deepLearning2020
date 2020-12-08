from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import tensorflow as image_summary
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm_notebook

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_img, train_lab = mnist.train.images, mnist.train.labels
test_img, test_lab = mnist.test.images, mnist.test.labels

EPOCH = 3
ITERATION = 5000
OUTPUT_PATH_LOG = './test'
L2_REGULARIZATION = False
BATCH_SIZE = 100


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x_in: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={
                      x_in: v_xs, y_in: v_ys, keep_prob: 1})
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
    units = sess.run(layer, feed_dict={xs: np.reshape(
        stimuli, [1, 784], order='F'), keep_prob: 1.0})
    plot_nn_filter(units)


def plot_nn_filter(units):
    import math
    filters = units.shape[3]
    plt.figure(1, figsize=(20, 20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")
    plt.tight_layout()
    plt.show()


tf.reset_default_graph()

x_in = tf.placeholder(dtype=tf.float32, shape=(None, 28*28))/255.
y_in = tf.placeholder(dtype=tf.float32, shape=(None, 10))

x_image = tf.reshape(x_in, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)

# 紀錄 3 張影像在 tensorboard 上
tf.summary.image('Input_images', x_image, max_outputs=3)
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
#     tf.summary.histogram('fc2/outputs', prediction)

## Loss ##
with tf.name_scope('loss'):
    l2_loss = 0.005*tf.nn.l2_loss(W_conv1) + 0.005*tf.nn.l2_loss(W_conv2) + 0.005*tf.nn.l2_loss(W_fc1) + 0.005*tf.nn.l2_loss(W_fc2)
    if L2_REGULARIZATION:
        loss = tf.reduce_mean(-tf.reduce_sum(y_in * tf.log(prediction) + l2_loss,
                                                reduction_indices=[1]))
    else:
        loss = tf.reduce_mean(-tf.reduce_sum(y_in * tf.log(prediction),
                                                reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

## optimizer ##
with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

result = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_in, 1))

# accuracy ##
with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(result, tf.float32))
    # 紀錄 accuracy
    tf.summary.scalar('Accuracy', accuracy)

# 把上面所有的 tf.summary 整合成一個 operation call
opsSummary = tf.summary.merge_all()

# images=x_image[:25]
# tf.summary.image('25 training data examples', images, max_outputs = 25)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Creat writer to write data needed for tensorboard
writer = tf.summary.FileWriter(OUTPUT_PATH_LOG, sess.graph)

# Show graph of neural network in tensorboard
writer.add_graph(sess.graph)

writerTrain = tf.summary.FileWriter(os.path.join(OUTPUT_PATH_LOG, 'train'))
writerTest = tf.summary.FileWriter(os.path.join(OUTPUT_PATH_LOG, 'test'))

sess.run(tf.global_variables_initializer())

for i in range(EPOCH):
    # mini batch training
    for j in tqdm_notebook(range(mnist.train.num_examples // BATCH_SIZE)):
        batch = mnist.train.next_batch(BATCH_SIZE)
        print(np.shape(batch[0]))
        print(np.shape(batch[1]))
        print(batch[1])
        sess.run(optimizer, feed_dict={
                 x_in: batch[0], y_in: batch[1], keep_prob: 0.5})

        if j % 50 == 0:
            print((i+1), ' epoch, ', j, ' iteration')
            train_accuracy = sess.run(accuracy, feed_dict={
                x_in: train_img[:500], y_in: train_lab[:500], keep_prob: 0})
            test_accuracy = sess.run(accuracy, feed_dict={
                x_in: test_img[:500], y_in: test_lab[:500],  keep_prob: 0})

            # print('Epoch %2d: acc = %.3f, test_acc = %.3f' %
            #     (i+1, train_accuracy*100., test_accuracy*100.))

            # Calculate desired information to be shown in tensorboard
            summary = sess.run(opsSummary, feed_dict={
                x_in: train_img[:500], y_in: train_lab[:500], keep_prob: 1})
            writerTrain.add_summary(summary, global_step=500*i+j)

            summary = sess.run(opsSummary, feed_dict={
                x_in: test_img[:500], y_in: test_lab[:500], keep_prob: 1})
            writerTest.add_summary(summary, global_step=500*i+j)

    # saver = tf.train.Saver()
    # saver.save(sess, os.path.join(OUTPUT_PATH_LOG, 'model.ckpt'))

    y_pre = sess.run(prediction, feed_dict={
        x_in: mnist.test.images[:1000], keep_prob: 1})
    prediction_array = tf.argmax(y_pre, 1).eval(
        session=tf.compat.v1.Session())
    print('predict number: \n', prediction_array)

    index = np.where(mnist.test.labels == 1)
    print('label number: \n', index[1][:1000])

    accuracy_num = compute_accuracy(
        mnist.test.images[:1000], mnist.test.labels[:1000])
    print('accuracy: ', accuracy_num)
