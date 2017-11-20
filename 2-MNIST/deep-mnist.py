import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

# Placeholders
# input image placeholder
x = tf.placeholder(tf.float32, shape=[None, 784])
# output label placeholder
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# HOW TO IMPROVE from 92%
# need to use WEIGHT INITIALIZATION
# initialize weights with small amount of noise for symmetry breaking and
# to prevent 0 gradients

# rather than doing by hand, use these functions instead:
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution and pooling
# used
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# set up first set of conv networks
# We can now implement our first layer. It will consist of convolution, followed by max pooling.
# The convolution will compute 32 features for each 5x5 patch.
# Its weight tensor will have a shape of [5, 5, 1, 32].
# The first two dimensions are the patch size,
# the next is the number of input channels, and the last is the number of output channels.
W_conv1 = weight_variable([5, 5, 1, 32])
# We will also have a bias vector with a component for each output channel.
b_conv1 = bias_variable([32])

# To apply the layer, we first reshape x to a 4d tensor,
# with the second and third dimensions corresponding to image width and height,
# and the final dimension corresponding to the number of color channels.
x_image = tf.reshape(x, [-1, 28, 28, 1])


# We then convolve x_image with the weight tensor,
# add the bias, apply the ReLU function, and finally max pool.
# The max_pool_2x2 method will reduce the image size to 14x14.

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# buiding deep neural network by stacking layers
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # function
    y = tf.matmul(x,W) + b

    # loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    for _ in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
