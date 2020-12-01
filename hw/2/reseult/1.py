from tensorflow.examples. tutorials.mnist import input_data

minst = input_data.read_data_sets(“MNIST_data/”,one_hot=True)
training_data = mnist.train.images
training_label = mnist.train.labels