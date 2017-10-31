import numpy as np
import tensorflow as tf
from utils import Utils as utils

class Svm:

    def assertions(self):
        assert True, "Dummy"

    def __init__(self, input_size, output_size, optimization_function='gradient-descent', learning_rate=0.01, learning_rate_decay='none', epochs=10, gamma=-10.0, batch_size=100, scope_name='default'):
        self.input_size = input_size
        self.output_size = output_size

        self.optimization_function = optimization_function
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.epochs = epochs
        self.gamma = gamma

        self.batch_size = batch_size
        self.scope_name = scope_name
        self.assertions()

        self.initial_learning_rate = self.learning_rate

        self._create_model()

    def _create_model(self):
        with tf.variable_scope(self.scope_name) as scope:
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size], name='x')
            self.y = tf.placeholder(dtype=tf.float32, shape=[self.output_size, None], name='y')
            self.prediction_grid = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)

            self.bias = tf.Variable(tf.random_normal(shape=[self.output_size, self.batch_size]))

            #Gaussian (RBF) kernel
            gamma = tf.constant(self.gamma)
            dist = tf.reshape(tf.reduce_sum(tf.square(self.x), 1), [-1,1])
            sq_dists = tf.multiply(2., tf.matmul(self.x, tf.transpose(self.x)))
            my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))        

            #SVM Model
            first_term = tf.reduce_sum(self.bias)
            b_vec_cross = tf.matmul(tf.transpose(self.bias), self.bias)
            y_target_cross = self._reshape_matmul(self.y)
            second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)), [1, 2])
            
            #Gaussian (RBF) prediction kernel
            rA = tf.reshape(tf.reduce_sum(tf.square(self.x), 1), [-1,1])
            rB = tf.reshape(tf.reduce_sum(tf.square(self.prediction_grid), 1), [-1,1])
            pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(self.x, tf.transpose(self.prediction_grid)))), tf.transpose(rB))
            pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

            prediction_output = tf.matmul(tf.multiply(self.y, self.bias), pred_kernel)
            self.predictions = tf.argmax(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)

            self.loss = tf.reduce_mean(tf.negative(tf.subtract(first_term, second_term)))
            self.optimizer = utils.get_optimizer(self.optimization_function, self.learning_rate).minimize(self.loss)

            self.labels = tf.argmax(self.y, 0)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(self.y, 0)), tf.float32))
            
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
                avg_accuracy = 0.
                self.learning_rate = utils.get_learning_rate(self.learning_rate_decay, self.initial_learning_rate, epoch)
                for i in range(batches_per_epoch):
                    batch_x, batch_y = utils.get_batch(X, Y, self.batch_size)
                    batch_y = utils.classes_matching_matrix(batch_y)
                    sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.prediction_grid: batch_x })
                    loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y, self.prediction_grid: batch_x })
                    avg_loss += loss
                    avg_accuracy += accuracy
                avg_loss /= batches_per_epoch
                avg_accuracy /= batches_per_epoch
                print('epoch {0}: loss = {1:.6f}, accuracy = {2:.2f}%'.format(epoch, avg_loss, avg_accuracy * 100))
            self.saver.save(sess, './weights/svm/' + self.scope_name + '/checkpoint', global_step=0)

    def test(self, X, Y):
        batches_per_epoch = int(len(X) / self.batch_size)

        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('./weights/svm/' + self.scope_name))
            avg_accuracy = 0.
            avg_loss = 0.
            counters = [[0 for i in range(self.output_size)] for j in range(self.output_size)]
            for i in range(batches_per_epoch):
                batch_x, batch_y = utils.get_sequential_batch(X, Y, i * self.batch_size, self.batch_size)  
                batch_y = utils.classes_matching_matrix(batch_y)
                loss, accuracy, testLogits, testLabels = sess.run([self.loss, self.accuracy, self.predictions, self.labels], feed_dict={self.x: batch_x, self.y: batch_y, self.prediction_grid: batch_x})            
                
                for i in range(len(testLabels)):
                    counters[testLabels[i]][testLogits[i]] += 1
                
                avg_loss += loss
                avg_accuracy += accuracy
            [print("class {0}, accuracy = {1:.2f}, values =".format(i+1, counters[i][i] / np.sum(counters[i])), counters[i]) for i in range(len(counters))]
            avg_loss /= batches_per_epoch
            avg_accuracy /= batches_per_epoch
            print("Test: loss = {0:.6f}, accuracy = {1:.2f}%".format(avg_loss, avg_accuracy * 100))

    def _reshape_matmul(self, mat):
        v1 = tf.expand_dims(mat, 1)
        v2 = tf.reshape(v1, [self.output_size, self.batch_size, 1])
        return(tf.matmul(v2, v1))