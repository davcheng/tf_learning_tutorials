# made by david for testing
import tensorflow as tf

# imagine
# [[0,1],[0,1]] is right
# [[1,1],[0,0]] is up
# [[1,0],[1,0]] is left
# [[0,0],[1,1]] is down

# placeholders
# x shape needs to be [rank, # of inputs]
x = tf.placeholder(tf.float32, shape=[None, 4], name="x")
# y_ shape needs to be [rank, # of possible outcomes (if using one-hot)]
y_ = tf.placeholder(tf.float32, [None,2], name="y_")

# Variables (for weights and bias), both initialized to zeros
# W shape needs to be zeros([ # of inputs used in x, # possible outcomes used in y_])
W = tf.Variable(tf.zeros([4, 2]), name="W")
# b shape needs to be zeros([# possible outcomes used in y_])
b = tf.Variable(tf.zeros([2]), name="b")

# specify linear model (equlivalent to W*x + b)
# y = tf.matmul(W,x) + b
y = tf.nn.softmax(tf.matmul(x, W) + b)
# cross_entropy == loss
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
squared_deltas = tf.square(y - y_)
loss = tf.reduce_sum(squared_deltas)

# optimizer
# this is an equlivalent train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # x_train = [height, weight, shoe size, BMI]
    x_train = [[2,190,11,21], [1,95,5,30], [1.2,100,6,31], [1.8,200,13,23], [1,105,7,25], [1.6,210,9,31]]
    y_train = [[1,0], [0,1], [0,1], [1,0], [0,1], [1,0]] # using one hot vector ([M,F], not prob -- boolean)
    # x_train = [[2,190,11], [1,95,5], [1.9,200,10], [1.3,120,7]]
    # y_train = [0,1,0,1]
    # y_train = [[.8,.2],[.2,.8]]
    for i in range(5):
        sess.run(train, {x: x_train, y_:y_train})
        curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y_: y_train})
        print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

# placeholders
# x = tf.placeholder(tf.float32, shape=[None, 4], name="x")
# y_ = tf.placeholder(tf.float32, shape=[None, 4], name="y_")
#
# # Variables (for weights and bias), both initialized to zeros
# W = tf.Variable(tf.zeros([4,4]), name="W")
# b = tf.Variable(tf.zeros([4]), name="b")
#
# # specify linear model (equlivalent to W*x + b)
# y = tf.matmul(W,x) + b
# # cross_entropy == loss
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#
# # optimizer
# # this is an equlivalent train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(cross_entropy)
#
# with tf.Session() as sess:
#
#     # x_train = [[1,1,0,0], [0,1,0,1], [0,0,1,1], [1,0,1,0], [1,1,0,0], [[0,1],[0,1]], [[0,0],[1,1]], [[1,0],[1,0]]]
#     # x_train = [([1,1,0,0]), ([0,1,0,1]), ([0,0,1,1]), ([1,0,1,0]),([1,1,0,0]), ([0,1,0,1]), ([0,0,1,1]), ([1,0,1,0]), ([1,1,0,0]), ([0,1,0,1]), ([0,0,1,1]), ([1,0,1,0])]
#     x_train = [[1,1,0,0], [0,1,0,1], [0,0,1,1], [1,0,1,0], [1,1,0,0], [0,1,0,1], [0,0,1,1], [1,0,1,0], [1,1,0,0], [0,1,0,1], [0,0,1,1], [1,0,1,0]]
#     y_train = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
#
#     for i in xrange(100):
#         sess.run(train, {x: x_train, y: y_train})
#         curr_W, curr_b, curr_loss = sess.run([W, b, cross_entropy], {x: x_train, y: y_train})
#         print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
#         # batch = dumbmnist.train.next_batch(100)
#         # train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#         # curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
#         # print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
#     print(sess.run([W, b]))


# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
