import numpy as np
import tensorflow as tf

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear', 'softplus']
allowed_losses = ['rmse', 'sigmoid-cross-entropy', 'softmax-cross-entropy']


class ForwardClassifier:
    """A deep autoencoder with denoising capability"""

    def assertions(self):
        global allowed_activations, allowed_noises, allowed_losses
        assert 'list' in str(
            type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.activations) == len(
            self.dims), "No. of activations must equal to no. of hidden layers"
        assert self.epoch > 0, "No. of epoch must be atleast 1"
        assert set(self.activations + allowed_activations) == set(
            allowed_activations), "Incorrect activation given."
        assert self.output_activation in allowed_activations, "Incorrect output activation given."
        assert self.loss in allowed_losses, "Incorrect loss given."

    def __init__(self, dims, activations, output_activation, loss, epoch=1000,
                 lr=0.001, batch_size=100, print_step=50):
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.output_activation = output_activation
        self.epoch = epoch
        self.dims = dims
        self.assertions()
        self.activations.append(self.output_activation)
        self.depth = len(dims)
        self.weights, self.biases = [], []

    def fit(self, data_x, data_y):
        tf.reset_default_graph()
        input_dim = len(data_x[0])
        output_dim = len(data_y[0])
        if(self.depth > 0):
            hidden_dim = self.dims[0]
        else:
            hidden_dim = output_dim

        sess = tf.Session()

        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        y = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name='y')

        weights, biases = [],[]

        for i in range(self.depth + 1):
            weights.append(tf.Variable(tf.truncated_normal([input_dim, hidden_dim], dtype=tf.float32)))
            biases.append(tf.Variable(tf.truncated_normal([hidden_dim],dtype=tf.float32)))

            input_dim = hidden_dim
            if(i < self.depth - 1):
                hidden_dim = self.dims[i+1]
            else:
                hidden_dim = output_dim

        status = x
        #Forward
        for i in range(self.depth + 1):
            status = self.activate(tf.matmul(status, weights[i]) + biases[i], self.activations[i])
        #status.append(self.activate(tf.matmul(status[self.depth-1], weights[self.depth]) + biases[self.depth], self.output_activation))

        # reconstruction loss
        if self.loss == 'rmse':
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, status))))
        elif self.loss == 'softmax-cross-entropy':
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=status, labels=y))
        elif self.loss == 'sigmoid-cross-entropy':
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=status, labels=y))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss) #TODO: Use also AdamOptimizer, GradientDescentOptimizer

        sess.run(tf.global_variables_initializer())#initialize_all_variables())
        for i in range(self.epoch):
            b_x, b_y = self.get_batch(data_x, data_y, self.batch_size)
            sess.run(train_op, feed_dict={x: b_x, y: b_y})
            if (i + 1) % self.print_step == 0:
                l = sess.run(loss, feed_dict={x: data_x, y: data_y})
                print('epoch {0}: global loss = {1}'.format(i, l))

        for i in range(self.depth + 1):
            self.weights.append(sess.run(weights[i]))
            self.biases.append(sess.run(biases[i]))

    def transform(self, data_x):
        tf.reset_default_graph()
        sess = tf.Session()
        x = tf.constant(data_x, dtype=tf.float32)
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)

        return x.eval(session=sess)

    def test(self, data_x, data_y, samples_shown=1):
        data_y_ = self.transform(data_x)
        errors = 0.0
        total = 0.0
        for y_, y in zip(data_y_, data_y):
            real_class = np.argmax(y)
            predicted_class = np.argmax(y_)
            if(real_class != predicted_class):
                errors += 1
            total += 1
        print('Accuracy: {0:.2f}% (Errors: {1:.0f}, Total: {2:.0f})'.format((1-errors/total)*100, errors, total))

    def fit_transform(self, data_x, data_y):
        self.fit(data_x, data_y)
        return self.transform(data_x)

    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'softplus':
            return tf.nn.softplus(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')

    def get_batch(self, X, Y, size):
        a = np.random.choice(len(X), size, replace=False)
        return X[a], Y[a]
