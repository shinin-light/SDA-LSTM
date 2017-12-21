import tensorflow as tf
import numpy as np
import sys
from utils import Utils as utils

class StackedAutoEncoder:
    
    def assertions(self):
        assert 'list' in str( type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.encoding_functions) == len(self.dims), "No. of activations must equal to no. of hidden layers"
        assert len(self.decoding_functions) == len( self.dims), "No. of decoding activations must equal to no. of hidden layers"
        assert len(self.loss_functions) == len(self.dims), "No. of loss functions must equal to no. of hidden layers"
        assert self.epochs > 0, "No. of epochs must be at least 1"
        assert utils.noise_validator(self.noise) == True, "Invalid noises."

    def __init__(self, printer, input_size, dims, encoding_functions, decoding_functions, loss_functions, optimization_function, noise, epochs=10,
                 learning_rate=0.001, learning_rate_decay='none', batch_size=100, scope_name='default', initialization_function='xavier', early_stop_lookahead=5):
        self.printer = printer
        self.input_size = input_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.initial_learning_rate = utils.get_learning_rate(self.learning_rate_decay, self.learning_rate, 0)
        self.loss_functions = loss_functions
        self.optimization_function = optimization_function
        self.encoding_functions = encoding_functions
        self.decoding_functions = decoding_functions
        self.noise = noise
        self.epochs = epochs
        self.dims = dims
        self.depth = len(dims)
        self.scope_name = scope_name
        self.early_stop_lookahead = early_stop_lookahead
        self.initialization_function = initialization_function
        self.weights, self.biases, self.decoding_biases = [], [], []
        self.assertions()
        self._create_model()
        
    def _create_model(self):
        with tf.variable_scope(self.scope_name) as scope:
            self.x = []
            for i in range(self.depth):
                if(i == 0):
                    self.x.append(tf.placeholder(tf.float32, [None, self.input_size]))
                else:
                    self.x.append(tf.placeholder(tf.float32, [None, self.dims[i - 1]]))
            
            self.y = []
            for i in range(self.depth):
                if(i == 0):
                    self.y.append(tf.placeholder(tf.float32, [None, self.input_size]))
                else:
                    self.y.append(tf.placeholder(tf.float32, [None, self.dims[i - 1]]))
            self.lr = tf.placeholder(tf.float32)

            self.weights, self.biases = [], [] #ENC1, ENC2, ENC3, DEC3, DEC2, DEC1
            self.layerwise_encoded = [] #ENC1(X), ENC2(X), ENC3(X)
            self.layerwise_decoded = [] #DEC1(ENC1(X)), DEC2(ENC2(X)), DEC3(ENC3(X))
            self.encoded = [] #ENC1(X), ENC2(ENC1(X)), ENC3(ENC2(ENC1(X))) => ENCODED
            self.decoded = [] #DEC3(ENCODED), DEC2(DEC3(ENCODED)), DEC1(DEC2(DEC3(ENCODED)))

            initializer = utils.get_initializer(self.initialization_function)

            previous_size = self.input_size
            for i in range(self.depth):
                hidden_size = self.dims[i]
                
                encoding_weights = tf.get_variable('w' + str(i), shape=[previous_size, hidden_size], initializer=initializer, dtype=tf.float32)
                encoding_biases = tf.get_variable('eb' + str(i), shape=[hidden_size], initializer=initializer, dtype=tf.float32)
                encoding_function = utils.get_activation(self.encoding_functions[i])

                self.weights.append(encoding_weights)
                self.biases.append(encoding_biases)
                
                if(len(self.encoded) == 0):
                    self.encoded.append(encoding_function(tf.matmul(self.x[0], encoding_weights) + encoding_biases))
                else:
                    self.encoded.append(encoding_function(tf.matmul(self.encoded[len(self.encoded) - 1], encoding_weights) + encoding_biases))

                self.layerwise_encoded.append(encoding_function(tf.matmul(self.x[i], encoding_weights) + encoding_biases))

                previous_size = hidden_size

            for i in range(self.depth - 1, -1, -1):
                hidden_size = self.dims[i - 1] if i > 0 else self.input_size

                decoding_weights = tf.transpose(self.weights[i])
                decoding_biases = tf.get_variable('db' + str(i), shape=[hidden_size], initializer=initializer, dtype=tf.float32)
                decoding_function = utils.get_activation(self.decoding_functions[i])

                self.weights.append(decoding_weights)
                self.biases.append(decoding_biases)

                output_activation = utils.get_output_activation(self.loss_functions[i])
                if(len(self.decoded) == 0):
                    self.decoded.append(output_activation(decoding_function(tf.matmul(self.encoded[self.depth - 1], decoding_weights) + decoding_biases)))
                else:
                    self.decoded.append(output_activation(decoding_function(tf.matmul(self.decoded[len(self.decoded) - 1], decoding_weights) + decoding_biases)))

                self.layerwise_decoded.append(output_activation(decoding_function(tf.matmul(self.layerwise_encoded[i], decoding_weights) + decoding_biases)))

            self.encoded_data = self.encoded[self.depth - 1]
            self.decoded_data = self.decoded[self.depth - 1]
            self.output = self.decoded_data

            self.layerwise_losses = []
            self.layerwise_optimizers = []
            for i in range(self.depth):
                loss = utils.get_sdae_loss(logits=self.layerwise_decoded[self.depth - 1 - i], labels=self.y[i], name=self.loss_functions[i])
                optimizer = utils.get_optimizer(name=self.optimization_function, learning_rate=self.lr).minimize(loss)
                self.layerwise_losses.append(loss)
                self.layerwise_optimizers.append(optimizer)

            self.finetuning_loss = utils.get_sdae_loss(logits=self.decoded_data, labels=self.y[0], name=self.loss_functions[0])
            self.finetuning_optimizer = utils.get_optimizer(name=self.optimization_function, learning_rate=self.lr).minimize(self.finetuning_loss)

            #Saver
            self.saver = tf.train.Saver()
            
            self.merged_summary = []
            finetuning_summary = []
            #Tensorboard
            for i in range(self.depth):
                wg = tf.summary.histogram("weights-gradient-" + str(i), tf.gradients(self.layerwise_losses[i], [self.weights[i]]))
                ebg = tf.summary.histogram("encoding-biases-gradient-" + str(i), tf.gradients(self.layerwise_losses[i], [self.biases[i]]))
                dbg = tf.summary.histogram("decoding-biases-gradient-" + str(i), tf.gradients(self.layerwise_losses[i], [self.biases[self.depth * 2 - 1 - i]]))
                
                finetuning_summary.append(tf.summary.histogram("weights-finetuning-gradient-" + str(i), tf.gradients(self.finetuning_loss, [self.weights[i]])))
                finetuning_summary.append(tf.summary.histogram("encoding-biases-finetuning-gradient-" + str(i), tf.gradients(self.finetuning_loss, [self.biases[i]])))
                finetuning_summary.append(tf.summary.histogram("decoding-biases-finetuning-gradient-" + str(i), tf.gradients(self.finetuning_loss, [self.biases[self.depth * 2 - 1 - i]])))
                
                w = tf.summary.histogram("weights-" + str(i), self.weights[i])
                eb = tf.summary.histogram("encoding-biases-" + str(i), self.biases[i])
                db = tf.summary.histogram("decoding-biases-" + str(i), self.biases[self.depth * 2 - 1 - i])
                self.merged_summary.append(tf.summary.merge([wg,ebg,dbg,w,eb,db]))

                finetuning_summary.append([w, eb, db])
            self.finetuning_merged_summary = tf.summary.merge(finetuning_summary)

            self.writer = tf.summary.FileWriter("C:\\Users\\danie\\Documents\\SDA-LSTM\\logs\\sdae", graph=tf.get_default_graph())

    def train(self, Y, X_VAL, epochs=None, debug=False):
        batches_per_epoch = int(len(Y) / self.batch_size)
        
        self.global_step = 0

        if epochs is None:
            epochs = self.epochs

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for layer in range(self.depth):
                last_loss = sys.maxsize
                lookahead_counter = 1

                self.printer.print('Layer {0}'.format(layer + 1))
                tmp = np.copy(Y)
                tmp = utils.add_noise(tmp, self.noise[layer])
                X = tmp
                for epoch in range(epochs):
                    avg_loss = 0.
                    self.learning_rate = utils.get_learning_rate(self.learning_rate_decay, self.initial_learning_rate, epoch)
                    for i in range(batches_per_epoch):
                        batch_x, batch_y = utils.get_batch(X, Y, self.batch_size)
                        sess.run(self.layerwise_optimizers[layer], feed_dict={self.x[layer]: batch_x, self.y[layer]: batch_y, self.lr: self.learning_rate})
                        if debug:
                            loss, summary = sess.run([self.layerwise_losses[layer], self.merged_summary[layer]], feed_dict={self.x[layer]: batch_x, self.y[layer]: batch_y, self.lr: self.learning_rate})
                            self.writer.add_summary(summary, global_step=self.global_step)
                        else:
                            loss = sess.run(self.layerwise_losses[layer], feed_dict={self.x[layer]: batch_x, self.y[layer]: batch_y, self.lr: self.learning_rate})
                        avg_loss += loss
                        
                        self.global_step += 1
                    avg_loss /= batches_per_epoch
                    self.printer.print("Epoch {0}: loss = {1:.6f}".format(epoch, avg_loss))
            
                    validation_loss = sess.run(self.layerwise_losses[layer], feed_dict={self.x[layer]: X_VAL, self.y[layer]: X_VAL})
                
                    if validation_loss < last_loss:
                        self.printer.print("Validation loss: {0}".format(validation_loss))
                        last_loss = validation_loss
                        #self.saver.save(sess, './weights/sdae/' + self.scope_name + '/checkpoint', global_step=0)
                        lookahead_counter = 1
                    else:
                        if lookahead_counter >= self.early_stop_lookahead: 
                            break
                        lookahead_counter += 1
                self.saver.save(sess, './weights/sdae/' + self.scope_name + '/checkpoint', global_step=0) 
                Y = sess.run(self.layerwise_encoded[layer], feed_dict={self.x[layer]: Y, self.lr: self.learning_rate})
                X_VAL = sess.run(self.layerwise_encoded[layer], feed_dict={self.x[layer]: X_VAL, self.lr: self.learning_rate})
            #self.saver.save(sess, './weights/sdae/' + self.scope_name + '/checkpoint', global_step=0)

    def finetune(self, Y, X_VAL, epochs=None, debug=False):
        self.printer.print('Fine Tuning')
        last_loss = sys.maxsize
        lookahead_counter = 1

        if epochs is None:
            epochs = self.epochs

        batches_per_epoch = int(len(Y) / self.batch_size)
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('./weights/sdae/' + self.scope_name))
            tmp = np.copy(Y)
            tmp = utils.add_noise(tmp, self.noise[0])
            X = tmp
            for epoch in range(epochs):
                avg_loss = 0.
                self.learning_rate = utils.get_learning_rate(self.learning_rate_decay, self.initial_learning_rate, epoch)
                for i in range(batches_per_epoch):
                    batch_x, batch_y = utils.get_batch(X, Y, self.batch_size)
                    sess.run(self.finetuning_optimizer, feed_dict={self.x[0]: batch_x, self.y[0]: batch_y, self.lr: self.learning_rate})
                    if debug:
                        loss, summary = sess.run([self.finetuning_loss, self.finetuning_merged_summary], feed_dict={self.x[0]: batch_x, self.y[0]: batch_y, self.lr: self.learning_rate})
                        self.writer.add_summary(summary, global_step=self.global_step)
                    else:
                        loss = sess.run(self.finetuning_loss, feed_dict={self.x[0]: batch_x, self.y[0]: batch_y, self.lr: self.learning_rate})
                    avg_loss += loss
                    self.global_step += 1
                avg_loss /= batches_per_epoch
                self.printer.print('epoch {0}: loss = {1:.6f}'.format(epoch, avg_loss))

                validation_loss = sess.run(self.finetuning_loss, feed_dict={self.x[0]: X_VAL, self.y[0]: X_VAL})
                
                if validation_loss < last_loss:
                    self.printer.print("Validation loss: {0}".format(validation_loss))
                    last_loss = validation_loss
                    #self.saver.save(sess, './weights/sdae/' + self.scope_name + '/checkpoint', global_step=0)
                    lookahead_counter = 1
                else:
                    if lookahead_counter >= self.early_stop_lookahead: 
                        break
                    lookahead_counter += 1
            self.saver.save(sess, './weights/sdae/' + self.scope_name + '/checkpoint', global_step=0)

    def encode(self, data):
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('./weights/sdae/' + self.scope_name))
            return sess.run(self.encoded_data, feed_dict={self.x[0]: data})

    def timeseries_encode(self, data):
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('./weights/sdae/' + self.scope_name))
            result = []
            for sequence in data:
                encoded_sequence = sess.run(self.encoded_data, feed_dict={self.x[0]: sequence})
                result.append(encoded_sequence)
            return np.array(result)

    def test(self, data, samples_shown=1, threshold=0.0):
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('./weights/sdae/' + self.scope_name))
            avg_loss, output = sess.run([self.finetuning_loss, self.output], feed_dict={self.x[0]: data})
            for i in np.random.choice(len(data), samples_shown):
                self.printer.print('Sample {0}'.format(i))
                for d, d_ in zip(data[i], output[i]):
                    if(abs(d-d_) >= threshold):
                        self.printer.print('\tOriginal: {0:.2f} --- Reconstructed: {1:.2f} --- Difference: {2:.2f}'.format(d,d_,d-d_))
            self.printer.print("Test: loss = {0:.6f}".format(avg_loss))