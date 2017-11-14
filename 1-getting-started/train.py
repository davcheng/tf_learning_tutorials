# basic using gradient descent
import tensorflow as tf

# placeholders
x = tf.placeholder(tf.float32, shape=[None], name="x")
y = tf.placeholder(tf.float32, shape=[None], name="y")

# using variables
W = tf.Variable([-.3], dtype=tf.float32, name="W") # initializes to .3
b = tf.Variable([.3], dtype=tf.float32, name="b") # initializes to -.3


linear_model = W*x + b

with tf.Session() as sess:
    # add line for using tensorboard
    # access tensorboard with:
    # tensorboard --logdir="./graphs" --port 6006
    writer = tf.summary.FileWriter("./graphs", sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    # fix manually by assigning
    # fixW = tf.assign(W, [-1])
    # fixb = tf.assign(b, [1])
    # sess.run([fixW, fixb])

    # # loss function
    # squared_deltas = tf.square(linear_model - y)
    # loss = tf.reduce_sum(squared_deltas)
    #
    # print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

    # loss function
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    x_train = [1, 2, 3, 4]
    y_train = [0,-1,-2,-3]

    for i in xrange(100):
        sess.run(train, {x: x_train, y: y_train})
        curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
        print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

    print(sess.run([W, b]))
    # evaluate training accuracy


writer.close() # close the writer when you're done using it
