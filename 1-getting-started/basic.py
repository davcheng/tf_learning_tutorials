# basic using tf for creating placeholders, variables, and constants
# using tensorboard to visualize and run
import tensorflow as tf

# initializing a constant
a = tf.constant(3.0, dtype=tf.float32)

with tf.Session() as sess:
    print(sess.run(a))

# using placeholders
b = tf.placeholder(tf.float32)
c = tf.placeholder(tf.float32)

with tf.Session() as sess:
    print(sess.run(a+b, {a:[3,1,4], b:[3,1,4]}))

# using variables
W = tf.Variable([.3], dtype=tf.float32, name="W")
b = tf.Variable([-.3], dtype=tf.float32, name="b")
x = tf.placeholder(tf.float32, name="x")

linear_model = W*x + b

with tf.Session() as sess:
    # add line for using tensorboard
    # access tensorboard with:
    # tensorboard --logdir="./graphs" --port 6006
    writer = tf.summary.FileWriter("./graphs", sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(linear_model, {x:[3,1,4]}))

writer.close() # close the writer when you're done using it
