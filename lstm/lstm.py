from __future__ import print_function
import numpy as np
import tensorflow as tf
import time
from tensorflow.contrib import rnn

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear', 'softplus']
allowed_losses = ['rmse', 'softmax-cross-entropy', 'sigmoid-cross-entropy']
allowed_initializers = ['uniform', 'xavier']
allowed_optimizers = ['gradient-descent','adam']

e_values1 = np.load("../data/e_records.npy")
e_classes1 = np.load("../data/e_classes.npy")
t_values1= np.load("../data/t_records.npy")
t_classes1 = np.load("../data/t_classes.npy")

e_values = e_values1[:,:-1,0:30]
e_classes = e_classes1[:,1:]
t_values = t_values1[:,:-1,0:30]
t_classes = t_classes1[:,1:]

def get_batches(X, X_, size): #X and X_ are tuples
    assert size > 0, "Size should positive"
    dim = len(X)
    batch = []
    batch_ = []
    sequence_number = []
    for i in range(dim):
        idx = np.random.choice(len(X[i]), size, replace=False)
        batch.append(X[i][idx])
        batch_.append(X_[i][idx])
        sequence_number.append(np.full(size, len(X[i][0])))
    return batch, batch_, sequence_number

def get_sequential_batches(X, X_, start, size): #X and X_ are tuples
    assert size > 0, "Size should positive"
    assert start >= 0, "Start should not be negative"
    dim = len(X)
    batch = []
    batch_ = []
    sequence_number = []
    for i in range(dim):
        batch.append(X[i][start:start+size])
        batch_.append(X_[i][start:start+size])
        sequence_number.append(np.full(size, len(X[i][0])))
    return batch, batch_, sequence_number

class Lstm:
    def assertions(self):
        global allowed_activations, allowed_losses
        assert self.activation_function in allowed_activations, "Incorrect activation given."
        assert self.loss_function in allowed_losses, "Incorrect loss given."
        assert self.initialization_function in allowed_initializers, "Incorrect initializer given."
        assert self.optimization_function in allowed_optimizers, "Incorrect optimizer given."

    def __init__(self, input_size, state_size, output_size, max_sequence_length, activation_function, loss_function, 
                initialization_function='uniform', optimization_function='gradient-descent', epoch=1000, learning_rate=0.01, batch_size=16):
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.max_sequence_length = max_sequence_length
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.initialization_function = initialization_function
        self.optimization_function = optimization_function
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.assertions()
        self._create_model()

    def _create_model(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size, None, self.input_size]) #batch - timeseries - input vector
        self.y = tf.placeholder(tf.float32, [self.batch_size, None, self.output_size]) #batch - timeseries - class vector
        self.sequence_length = tf.placeholder(tf.int32, [self.batch_size]) #batch - timeseries length TODO is it ok?
        initializer = self.get_initializater(self.initialization_function)
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.state_size, num_proj=self.output_size, initializer=initializer) #TODO check if all the gates are present

        activation = self.get_activation(self.activation_function)

        outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.x, sequence_length=self.sequence_length, dtype=tf.float32)

        self.loss = self.get_loss(logits=outputs, labels=self.y, name=self.loss_function)
        self.optimizer = self.get_optimizer(name=self.optimization_function, learning_rate=self.learning_rate, loss=self.loss)

        #Saver
        self.saver = tf.train.Saver()
        
        #Tensorboard
        #writer = tf.summary.FileWriter("C:\\Users\\danie\\Documents\\SDA-LSTM\\logs", graph=tf.get_default_graph())
    
    def train(self, X, Y):
        train_size = len(X[0])
        train_dataset_num = len(X)
        batches_per_epoch = int(train_size / (self.batch_size * train_dataset_num))
        initial_learning_rate = self.learning_rate

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epoch):
                avg_loss = 0.
                self.learning_rate = initial_learning_rate / (10 * (epoch + 1))
                for i in range(batches_per_epoch):
                    batches_x, batches_y, batches_length = get_batches(X, Y, self.batch_size)
                    for dataset in range(train_dataset_num):
                        sess.run(self.optimizer, feed_dict={self.x: batches_x[dataset], self.y: batches_y[dataset], self.sequence_length: batches_length[dataset]})
                        loss = self.loss.eval(feed_dict={self.x: batches_x[dataset], self.y: batches_y[dataset], self.sequence_length: batches_length[dataset]})
                        avg_loss += loss
                avg_loss /= (batches_per_epoch * train_dataset_num)
                print("Epoch {0}, loss = {1}".format(epoch, avg_loss))
            self.saver.save(sess, './checkpoint',0)
    
    def test_loss(self, X, Y):
        train_size = len(X[0])
        train_dataset_num = len(X)
        batches_per_epoch = int(train_size / (self.batch_size * train_dataset_num))
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('.'))
            avg_loss = 0.
            for i in range(batches_per_epoch):
                batches_x, batches_y, batches_length = get_sequential_batches(X, Y, i * self.batch_size, self.batch_size)
                for dataset in range(train_dataset_num):
                    loss = self.loss.eval(feed_dict={self.x: batches_x[dataset], self.y: batches_y[dataset], self.sequence_length: batches_length[dataset]})
                    avg_loss += loss
            avg_loss /= (train_dataset_num * batches_per_epoch)
            print("Test global loss = {0}".format(avg_loss))
        
    def get_activation(self, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid
        elif name == 'softmax':
            return tf.nn.softmax
        elif name == 'softplus':
            return tf.nn.softplus
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh
        elif name == 'relu':
            return tf.nn.relu

    def get_loss(self, logits, labels, name):
        if name == 'rmse':
            return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, logits))))
        elif name == 'softmax-cross-entropy':
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        elif name == 'sigmoid-cross-entropy':
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    def get_initializater(self, name):
        if name == 'uniform':
            return tf.random_uniform_initializer(-1, 1)
        elif name == 'xavier':
            return tf.contrib.layers.xavier_initializer()

    def get_optimizer(self, name, learning_rate, loss):
        if name == 'gradient-descent':
            return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
        elif name == 'adam':
            return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

lstm = Lstm(input_size=30, state_size=2, output_size=20, max_sequence_length=6, activation_function='softmax', 
            loss_function='softmax-cross-entropy', initialization_function='uniform', optimization_function='gradient-descent',
            learning_rate=0.1, batch_size=64, epoch=30)
            
selection = np.random.choice(len(e_values), min(len(t_values),len(e_values)), replace=False)
e_values = e_values[selection]
e_classes = e_classes[selection]
idx = np.random.rand(len(e_values)) < 0.8
lstm.train([e_values[idx],t_values[idx]], [e_classes[idx],t_classes[idx]])
lstm.test([e_values[~idx],t_values[~idx]], [e_classes[~idx],t_classes[~idx]])