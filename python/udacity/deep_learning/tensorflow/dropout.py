# Solution is available in the other "solution.py" tab
import tensorflow as tf

output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])

# TODO: Create Model
keep_prob = tf.placeholder(tf.float32) # probability to keep units

# Hidden Layer with ReLU activation function
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)

# Dropout
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])


# TODO: Print session results
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(logits, feed_dict={keep_prob: 0.5}))

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for epoch_i in range(epochs):
#         for batch_i in range(batches):
#
#             sess.run(logits, feed_dict={
#                 features: batch_features,
#                 labels: batch_labels,
#                 keep_prob: 0.5})
#
#     validation_accuracy = sess.run(accuracy, feed_dict={
#         features: test_features,
#         labels: test_labels,
#         keep_prob: 0.5})