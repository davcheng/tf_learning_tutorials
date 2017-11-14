import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
dumbmnist = input_data.

sess = tf.InteractiveSession()

# Placeholders
# input image placeholder
x = tf.placeholder(float32, shape=[None, 784])
# output label placeholder
y = tf.placeholder(float32, shape=[None, 10])

# Variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

# loss function
y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
