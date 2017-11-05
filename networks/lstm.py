import numpy as np
import tensorflow as tf
from utils import Utils as utils
from tensorflow.contrib import rnn

class Lstm:
    def assertions(self):
        global allowed_activations, allowed_losses
        assert self.max_sequence_length > 0, "Incorrect max sequence length"
        assert utils.noise_validator(self.noise) == True, "Invalid noise."
        #assert self.cost_mask.shape[0] == self.output_size, "Invalid cost mask length."

    def __init__(self, max_sequence_length, input_size, state_size, output_size, loss_function, accuracy_function, activation_function='tanh',
                initialization_function='uniform', optimization_function='gradient-descent', epochs=10, learning_rate=0.01, 
                learning_rate_decay='none', noise='none', batch_size=16, cost_mask=np.array([]), scope_name='default'):
        self.max_sequence_length = max_sequence_length
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.accuracy_function = accuracy_function
        self.initialization_function = initialization_function
        self.optimization_function = optimization_function
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.initial_learning_rate = utils.get_learning_rate(self.learning_rate_decay, self.learning_rate, 0)
        self.noise = noise
        self.batch_size = batch_size
        self.scope_name = scope_name
        if(not len(cost_mask) > 0): #TODO handle output_size <= 0
            cost_mask = np.ones(self.output_size, dtype=np.float32)        
        self.cost_mask = tf.reshape(tf.constant(np.tile(cost_mask, batch_size * max_sequence_length), dtype=tf.float32), (batch_size, max_sequence_length, output_size))
        self.assertions()
        self._create_model()

    def _create_model(self):
        with tf.variable_scope(self.scope_name) as scope:
            self.x = tf.placeholder(tf.float32, [self.batch_size, self.max_sequence_length, self.input_size]) #batch - timeseries - input vector
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.max_sequence_length, self.output_size]) #batch - timeseries - class vector
            self.sequence_length = tf.placeholder(tf.int32, [self.batch_size])
            initializer = utils.get_initializater(self.initialization_function)
            activation = utils.get_activation(self.activation_function)
        
            cell = tf.nn.rnn_cell.LSTMCell(num_units=self.state_size, num_proj=self.output_size, initializer=initializer) #TODO check if all the gates are present

            outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.x, sequence_length=self.sequence_length, dtype=tf.float32)
            output_activation = utils.get_output_activation(self.loss_function)
            self.output = output_activation(outputs)

            self.loss = utils.get_one_hot_loss(logits=outputs, labels=self.y, name=self.loss_function, cost_mask=self.cost_mask)
            self.optimizer = utils.get_optimizer(name=self.optimization_function, learning_rate=self.learning_rate).minimize(self.loss)

            self.accuracy = utils.get_accuracy(logits=self.output, labels=self.y, name=self.accuracy_function)
        
            #Test
            self.testLogits = tf.reshape(self.output, (-1, self.output_size))
            self.testLabels = tf.reshape(self.y, (-1, self.output_size))

            #Saver
            self.saver = tf.train.Saver()
        
            #Tensorboard
            #writer = tf.summary.FileWriter("C:\\Users\\danie\\Documents\\SDA-LSTM\\logs", graph=tf.get_default_graph())
    
    def train(self, X, Y, lengths, epochs=None):
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
                    batch_x, batch_y, batch_length = utils.get_rnn_sequential_batch(X, Y, lengths, i * self.batch_size, self.batch_size)
                    batch_x = utils.add_noise(batch_x, self.noise)
                    sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.sequence_length: batch_length})
                    loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y, self.sequence_length: batch_length})
                    avg_loss += loss
                    avg_accuracy += accuracy
                avg_loss /= batches_per_epoch
                avg_accuracy /= batches_per_epoch
                print("Epoch {0}: loss = {1:.6f}, accuracy = {2:.2f}%".format(epoch, avg_loss, avg_accuracy * 100))
            self.saver.save(sess, './weights/lstm/' + self.scope_name + '/checkpoint', global_step=0)
    
    def test(self, X, Y, lengths):
        batches_per_epoch = int(len(X) / self.batch_size)

        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('./weights/lstm/' + self.scope_name))
            avg_accuracy = 0.
            avg_loss = 0.
            self.confusion_matrix = [[0 for i in range(self.output_size)] for j in range(self.output_size)]
            for i in range(batches_per_epoch):
                batch_x, batch_y, batch_length = utils.get_rnn_sequential_batch(X, Y, lengths, i * self.batch_size, self.batch_size)  
                loss, accuracy, testLogits, testLabels = sess.run([self.loss, self.accuracy, self.testLogits, self.testLabels], feed_dict={self.x: batch_x, self.y: batch_y, self.sequence_length: batch_length})            
                
                for i in range(len(testLabels)):
                    if np.sum(testLabels[i]) > 0:
                        label_idx = np.argmax(testLabels[i])
                        logit_idx = np.argmax(testLogits[i])
                        self.confusion_matrix[label_idx][logit_idx] += 1
                
                avg_loss += loss
                avg_accuracy += accuracy
            [print("class {0}, accuracy = {1:.2f}, values =".format(i+1, self.confusion_matrix[i][i] / np.sum(self.confusion_matrix[i])), self.confusion_matrix[i]) for i in range(len(self.confusion_matrix))]
            avg_loss /= batches_per_epoch
            avg_accuracy /= batches_per_epoch
            print("Test: loss = {0:.6f}, accuracy = {1:.2f}%".format(avg_loss, avg_accuracy * 100))
        return self.confusion_matrix

