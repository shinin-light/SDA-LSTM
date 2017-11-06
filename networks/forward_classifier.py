import numpy as np
import tensorflow as tf
from utils import Utils as utils

class ForwardClassifier:

    def assertions(self):
        assert 'list' in str(type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.activation_functions) == len(self.dims), "No. of activations must equal to no. of hidden layers"
        assert self.epochs > 0, "No. of epochs must be at least 1"

    def __init__(self, input_size, output_size, dims, activation_functions, loss_function, metric_function, optimization_function='gradient-descent', epochs=10,
                 learning_rate=0.001, learning_rate_decay='none', batch_size=100, cost_mask=np.array([]), scope_name='default'):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.initial_learning_rate = utils.get_learning_rate(self.learning_rate_decay, self.learning_rate, 0)
        self.loss_function = loss_function
        self.metric_function = metric_function
        self.optimization_function = optimization_function
        self.activation_functions = activation_functions
        self.epochs = epochs
        self.dims = dims
        self.scope_name = scope_name
        if(not len(cost_mask) > 0): #TODO handle output_size <= 0
            cost_mask = np.ones(self.output_size, dtype=np.float32)  
        self.cost_mask = tf.constant(cost_mask, dtype=tf.float32)
        self.assertions()
        self.activation_functions.append('linear')
        self.depth = len(dims)
        self.weights, self.biases = [], []
        self._create_model()

    def _create_model(self):
        with tf.variable_scope(self.scope_name) as scope:
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
            output_activation = utils.get_output_activation(self.loss_function)
            self.output = output_activation(outputs)

            self.loss = utils.get_one_hot_loss(logits=outputs, labels=self.y, name=self.loss_function, cost_mask=self.cost_mask)
            self.optimizer = utils.get_optimizer(name=self.optimization_function, learning_rate=self.learning_rate).minimize(self.loss)

            self.metric = utils.get_metric(logits=self.output, labels=self.y, name=self.metric_function)
            
            #Test
            self.testLogits = self.output
            self.testLabels = self.y

            #Saver
            self.saver = tf.train.Saver()

            #Tensorboard
            #writer = tf.summary.FileWriter("C:\\Users\\danie\\Documents\\SDA-LSTM\\logs", graph=tf.get_default_graph())

    def train(self, X, Y, epochs=None):
        batches_per_epoch = int(len(X) / self.batch_size)

        if epochs is None:
            epochs = self.epochs

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                avg_loss = 0.
                avg_metric = 0.
                self.learning_rate = utils.get_learning_rate(self.learning_rate_decay, self.initial_learning_rate, epoch)
                for i in range(batches_per_epoch):
                    batch_x, batch_y = utils.get_batch(X, Y, self.batch_size)
                    sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})
                    loss, metric = sess.run([self.loss, self.metric], feed_dict={self.x: batch_x, self.y: batch_y})
                    avg_loss += loss
                    avg_metric += metric
                avg_loss /= batches_per_epoch
                avg_metric /= batches_per_epoch
                print('epoch {0}: loss = {1:.6f}, metric ({2}) = {3:.6f}'.format(epoch, avg_loss, self.metric_function, avg_metric))
            self.saver.save(sess, './weights/forward/' + self.scope_name + '/checkpoint', global_step=0)

    def test(self, X, Y, samples_shown=1):
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('./weights/forward/' + self.scope_name))
            avg_loss, labels, logits = sess.run([self.loss, self.testLabels, self.testLogits], feed_dict={self.x: X, self.y: Y})

            print("Test: loss = {0:.6f}".format(avg_loss))

            metrics = utils.get_all_metrics(logits=np.array(logits), labels=np.array(labels))
            for m in metrics:
                print('\t', m, ':', metrics[m])

        return metrics