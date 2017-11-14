## Gather MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf


# x is placeholder;
x = tf.placeholder(tf.float32, shape=[None, 784])

# w are the weights; 784-dimensional image vectors
# to produce 10-dimensional vectors of evidence for the difference classes (0-9)
W = tf.Variable(tf.zeros([784, 10]))
# b is bias (bias for each class)
b = tf.Variable(tf.zeros([10]))

# Define model:
y = tf.nn.softmax(tf.matmul(x, W) + b)


# Now to train...
# what is the loss function?
#One very common, very nice function to determine the loss of a model is called "cross-entropy."
# Cross-entropy arises from thinking about information compressing codes in information theory
# but it winds up being an important idea in lots of areas,
# from gambling to machine learning. It's defined as:

# define placeholder to hold correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

# cross_entropy function: -Sigma(y'*log(y))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# use GradientDescentOptimizer to minimize cross_entropy
# learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# use InteractiveSession instead of tf.Session()

# with tf.InteractiveSession() as sess:
sess = tf.InteractiveSession()
# initializer
tf.global_variables_initializer().run()


print(sess.run(b))

# train for 1000 steps
for _ in range(1000):
    # each step, we get a random batch of 100 digits from training set
    # This is called STOCHASTIC TRAINING - when only using a small sample of training data
    # would use all for better results, but it is expensive
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # feed train_step in to replace the placeholders for x
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# evaluate or model
# checks if tf.argmax(y,1) is the label our model thinks is most likely for each input
# and sees if it is ACTUALLY equal to the TRUE label (y_)
# this will result in a list of booleans: [True, False, True, True]
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# use tf.reduce mean to covert [True, False, True, True] into an accuracy rate (92%)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
