import tensorflow as tf
import numpy as np
from utils import Utils as utils

class StackedAutoEncoder:
    
    def assertions(self):
        global allowed_activations, allowed_noises, allowed_losses
        assert 'list' in str( type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.epoch) == len(self.dims), "No. of epochs must equal to no. of hidden layers"
        assert len(self.encoding_functions) == len(self.dims), "No. of activations must equal to no. of hidden layers"
        assert len(self.decoding_functions) == len( self.dims), "No. of decoding activations must equal to no. of hidden layers"
        assert len(self.loss_functions) == len(self.dims), "No. of loss functions must equal to no. of hidden layers"
        assert all(True if x > 0 else False for x in self.epoch), "No. of epoch must be at least 1"

    def __init__(self, input_size, output_size, dims, encoding_functions, decoding_functions, loss_functions, optimization_function, noise, epoch=1000,
                 learning_rate=0.001, batch_size=100, print_step=50):
        self.input_size = input_size
        self.output_size = output_size
        self.print_step = print_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_functions = loss_functions
        self.optimization_function = optimization_function
        self.encoding_functions = encoding_functions
        self.decoding_functions = decoding_functions
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.depth = len(dims)
        self.weights, self.biases, self.decoding_biases = [], [], []
        self.assertions()
        self._create_model()
        
    def _create_model(self):
        self.x = []
        for i in range(self.depth):
            if(i == 0):
                self.x.append(tf.placeholder(tf.float32, [None, self.input_size]))
            else:
                self.x.append(tf.placeholder(tf.float32, [None, self.dims[i - 1]]))
        
        self.weights, self.biases = [], []
        self.layerwise_encoded = []
        self.layerwise_decoded = []
        self.encoded = []
        self.decoded = []

        previous_size = self.input_size
        for i in range(self.depth):
            hidden_size = self.dims[i]

            encoding_weights = tf.Variable(tf.truncated_normal([previous_size, hidden_size], dtype=tf.float32))
            encoding_biases = tf.Variable(tf.truncated_normal([hidden_size], dtype=tf.float32))
            encoding_function = utils.get_activation(self.encoding_functions[i])

            self.weights.append(encoding_weights)
            self.biases.append(encoding_biases)
            
            if(len(self.encoded) == 0):
                self.encoded.append(encoding_function(tf.matmul(self.x[0], encoding_weights) + encoding_biases))
            else:
                self.encoded.append(encoding_function(tf.matmul(self.encoded[len(self.encoded) - 1], encoding_weights) + encoding_biases))

            self.layerwise_encoded.append(encoding_function(tf.matmul(self.x[i], encoding_weights) + encoding_biases))

            previous_size = hidden_size

        previous_size = self.dims[self.depth - 1]
        for i in range(self.depth - 1, -1, -1):
            hidden_size = self.dims[i - 1] if i > 0 else self.input_size

            decoding_weights = tf.transpose(self.weights[i])
            decoding_biases = tf.Variable(tf.truncated_normal([hidden_size], dtype=tf.float32))
            decoding_function = utils.get_activation(self.decoding_functions[i])

            self.weights.append(decoding_weights)
            self.biases.append(decoding_biases)

            if(len(self.decoded) == 0):
                self.decoded.append(decoding_function(tf.matmul(self.encoded[self.depth - 1], decoding_weights) + decoding_biases))
            else:
                self.decoded.append(decoding_function(tf.matmul(self.decoded[len(self.decoded) - 1], decoding_weights) + decoding_biases))

            self.layerwise_decoded.append(decoding_function(tf.matmul(self.layerwise_encoded[i], decoding_weights) + decoding_biases))

            previous_size = hidden_size

        self.output = self.decoded[self.depth - 1]

        self.layerwise_losses = []
        self.layerwise_optimizers = []
        for i in range(self.depth):
            loss = utils.get_loss(logits=self.layerwise_decoded[self.depth -1 - i], labels=self.x[i], name=self.loss_functions[i])
            optimizer = utils.get_optimizer(name=self.optimization_function, learning_rate=self.learning_rate).minimize(loss)
            self.layerwise_losses.append(loss)
            self.layerwise_optimizers.append(optimizer)

        self.finetuning_loss = utils.get_loss(logits=self.x[0], labels=self.output, name=self.loss_functions[0])
        self.finetuning_optimizer = utils.get_optimizer(name=self.optimization_function, learning_rate=self.learning_rate).minimize(self.finetuning_loss)

        #Saver
        self.saver = tf.train.Saver()
        
        #Tensorboard
        #writer = tf.summary.FileWriter("C:\\Users\\danie\\Documents\\SDA-LSTM\\logs", graph=tf.get_default_graph())

    def train(self, X):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for layer in range(self.depth):
                tmp = np.copy(X)
                tmp = self._add_noise(tmp, layer)
                X = tmp
                print("Layer {0}:".format(layer + 1))
                print(len(X[0]))
                for i in range(self.epoch[layer]):
                    batch_x, batch_y = utils.get_batch(X, X, self.batch_size)
                    sess.run(self.layerwise_optimizers[layer], feed_dict={self.x[layer]: batch_x})
                    if (i + 1) % self.print_step == 0:
                        loss = sess.run(self.layerwise_losses[layer], feed_dict={self.x[layer]: X})
                        print('epoch {0}: global loss = {1}'.format(i, loss))
                X = sess.run(self.layerwise_encoded[layer], feed_dict={self.x[layer]: X})
                print(len(X[0]))
            self.saver.save(sess, './weights/sdae/checkpoint',0)

    def finetune(self, X):
        print('Fine Tuning')
        tmp = np.copy(X)
        tmp = self._add_noise(tmp, 0)
        X = tmp
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('./weights/sdae'))
            for i in range(self.epoch[0]):
                batch_x, batch_y = utils.get_batch(X, X, self.batch_size)
                sess.run(self.finetuning_optimizer, feed_dict={self.x[0]: batch_x})
                if (i + 1) % self.print_step == 0:
                    loss = sess.run(self.finetuning_loss, feed_dict={self.x[0]: X})
                    print('epoch {0}: global loss = {1}'.format(i, loss))
            self.saver.save(sess, './weights/sdae/checkpoint',0)

    def _add_noise(self, x, layer):
        if self.noise[layer] == 'none':
            return x
        if self.noise[layer] == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if 'mask' in self.noise[layer]:
            frac = float(self.noise[layer].split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), int(round(frac * len(i))), replace=False)
                i[n] = 0
            return temp
        if self.noise[layer] == 'sp':
            pass