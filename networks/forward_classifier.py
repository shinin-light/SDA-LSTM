import numpy as np
import tensorflow as tf
from utils import Utils as utils

class ForwardClassifier:

    def assertions(self):
        assert 'list' in str(type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.activation_functions) == len(self.dims), "No. of activations must equal to no. of hidden layers"
        assert self.epoch > 0, "No. of epoch must be at least 1"

    def __init__(self, input_size, output_size, dims, activation_functions, output_activation_function, loss_function, optimization_function='gradient-descent', epoch=1000,
                 learning_rate=0.001, batch_size=100, print_step=50):
        self.input_size = input_size
        self.output_size = output_size
        self.print_step = print_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.optimization_function = optimization_function
        self.output_activation_function = output_activation_function
        self.activation_functions = activation_functions
        self.epoch = epoch
        self.dims = dims
        self.assertions()
        self.activation_functions.append(self.output_activation_function)
        self.depth = len(dims)
        self.weights, self.biases = [], []
        self._create_model()

    def _create_model(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size], name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.output_size], name='y')
        
        if(self.depth > 0):
            hidden_size = self.dims[0]
        else:
            hidden_size = output_size
        previous_size = self.input_size

        self.weights, self.biases = [],[]

        for i in range(self.depth + 1):
            self.weights.append(tf.Variable(tf.truncated_normal([previous_size, hidden_size], dtype=tf.float32)))
            self.biases.append(tf.Variable(tf.truncated_normal([hidden_size],dtype=tf.float32)))

            previous_size = hidden_size
            if(i < self.depth - 1):
                hidden_size = self.dims[i+1]
            else:
                hidden_size = self.output_size

        outputs = self.x
        for i in range(self.depth + 1):
            activation = utils.get_activation(self.activation_functions[i])
            outputs = activation(tf.matmul(outputs, self.weights[i]) + self.biases[i])
        
        self.output = outputs

        self.loss = utils.get_loss(logits=outputs, labels=self.y, name=self.loss_function)
        self.optimizer = utils.get_optimizer(name=self.optimization_function, learning_rate=self.learning_rate).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #Saver
        self.saver = tf.train.Saver()

        #Tensorboard
        #writer = tf.summary.FileWriter("C:\\Users\\danie\\Documents\\SDA-LSTM\\logs", graph=tf.get_default_graph())

    def train(self, X, Y):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                batch_x, batch_y = utils.get_batch(X, Y, self.batch_size)
                sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})
                if (i + 1) % self.print_step == 0:
                    l = sess.run(self.loss, feed_dict={self.x: X, self.y: Y})
                    print('epoch {0}: global loss = {1}'.format(i, l))
            self.saver.save(sess, './weights/classifier/checkpoint',0)

    def test(self, X, Y, samples_shown=1):
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('./weights/classifier'))
            avg_loss, avg_accuracy = sess.run([self.loss, self.accuracy], feed_dict={self.x: X, self.y: Y})
            print("Test: loss = {0:.6f}, accuracy = {1:.2f}%".format(avg_loss, avg_accuracy * 100))