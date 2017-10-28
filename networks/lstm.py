import numpy as np
import tensorflow as tf
from utils import Utils as utils
from tensorflow.contrib import rnn

class Lstm:
    def assertions(self):
        global allowed_activations, allowed_losses
        assert self.max_sequence_length > 0, "Incorrect max sequence length"
        #assert self.cost_mask.shape[0] == self.output_size, "Invalid cost mask length."

    def __init__(self, max_sequence_length, input_size, state_size, output_size, loss_function, activation_function='tanh',
                initialization_function='uniform', optimization_function='gradient-descent', epoch=1000, learning_rate=0.01, batch_size=16, cost_mask=np.array([])):
        self.max_sequence_length = max_sequence_length
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.initialization_function = initialization_function
        self.optimization_function = optimization_function
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        if(not len(cost_mask) > 0 or not self.output_size > 0): #TODO handle output_size <= 0
            cost_mask = np.ones(self.output_size)        
        self.cost_mask = tf.reshape(tf.constant(np.tile(cost_mask, batch_size * max_sequence_length), dtype=tf.float32), (batch_size, max_sequence_length, output_size))
        self.assertions()
        self._create_model()

    def _create_model(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.max_sequence_length, self.input_size]) #batch - timeseries - input vector
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.max_sequence_length, self.output_size]) #batch - timeseries - class vector
        self.sequence_length = tf.placeholder(tf.int32, [self.batch_size]) #batch - timeseries length TODO is it ok?
        initializer = utils.get_initializater(self.initialization_function)
        activation = utils.get_activation(self.activation_function)
        
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.state_size, num_proj=self.output_size, initializer=initializer, activation=activation) #TODO check if all the gates are present

        outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.x, sequence_length=self.sequence_length, dtype=tf.float32)

        self.loss = utils.get_loss(logits=outputs, labels=self.y, name=self.loss_function, lengths=self.sequence_length, cost_mask=self.cost_mask)
        self.optimizer = utils.get_optimizer(name=self.optimization_function, learning_rate=self.learning_rate).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(outputs, 2), tf.argmax(self.y, 2))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #Test
        self.testLogits = tf.reshape(outputs, (-1, self.output_size))
        self.testLabels = tf.reshape(self.y, (-1, self.output_size))

        #Saver
        self.saver = tf.train.Saver()
        
        #Tensorboard
        #writer = tf.summary.FileWriter("C:\\Users\\danie\\Documents\\SDA-LSTM\\logs", graph=tf.get_default_graph())
    
    def train(self, X, Y, lengths):
        batches_per_epoch = int(len(X) / self.batch_size)
        initial_learning_rate = self.learning_rate

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epoch):
                avg_loss = 0.
                avg_accuracy = 0.
                self.learning_rate = (initial_learning_rate * 10) / (10 * (epoch + 1))
                for i in range(batches_per_epoch):
                    batches_x, batches_y, batches_length = utils.get_rnn_sequential_batch(X, Y, lengths, i * self.batch_size, self.batch_size)
                    sess.run(self.optimizer, feed_dict={self.x: batches_x, self.y: batches_y, self.sequence_length: batches_length})
                    loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict={self.x: batches_x, self.y: batches_y, self.sequence_length: batches_length})
                    avg_loss += loss
                    avg_accuracy += accuracy
                avg_loss /= batches_per_epoch
                avg_accuracy /= batches_per_epoch
                print("Epoch {0}: loss = {1:.6f}, accuracy = {2:.2f}%".format(epoch, avg_loss, avg_accuracy * 100))
            self.saver.save(sess, './checkpoint',0)
    
    def test(self, X, Y, lengths):
        batches_per_epoch = int(len(X) / self.batch_size)

        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('.'))
            avg_accuracy = 0.
            avg_loss = 0.
            counters = [[0 for i in range(self.output_size)] for j in range(self.output_size)]
            for i in range(batches_per_epoch):
                batches_x, batches_y, batches_length = utils.get_rnn_sequential_batch(X, Y, lengths, i * self.batch_size, self.batch_size)  
                loss, accuracy, testLogits, testLabels = sess.run([self.loss, self.accuracy, self.testLogits, self.testLabels], feed_dict={self.x: batches_x, self.y: batches_y, self.sequence_length: batches_length})            
                
                for i in range(len(testLabels)):
                        label_idx = np.argmax(testLabels[i])
                        logit_idx = np.argmax(testLogits[i])
                        counters[label_idx][logit_idx] += 1
                
                avg_loss += loss
                avg_accuracy += accuracy
            [print("class {0}, accuracy = {1:.2f}, values =".format(i+1, counters[i][i] / np.sum(counters[i])), counters[i]) for i in range(len(counters))]
            avg_loss /= batches_per_epoch
            avg_accuracy /= batches_per_epoch
            print("Test: loss = {0:.6f}, accuracy = {1:.2f}%".format(avg_loss, avg_accuracy * 100))

