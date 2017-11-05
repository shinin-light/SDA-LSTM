import numpy as np
import tensorflow as tf
import math

class Utils:

    def get_activation(name):
        if name == 'sigmoid':
            return tf.nn.sigmoid
        elif name == 'softmax':
            return tf.nn.softmax
        elif name == 'softplus':
            return tf.nn.softplus
        elif name == 'linear':
            return lambda x: x
        elif name == 'tanh':
            return tf.nn.tanh
        elif name == 'relu':
            return tf.nn.relu
        raise BaseException("Invalid activation function.")

    def get_output_activation(loss_function_name):
        if 'sigmoid' in loss_function_name:
            return tf.nn.sigmoid
        elif 'softmax' in loss_function_name:
            return tf.nn.softmax
        return lambda x: x

    def get_accuracy(logits, labels, name):
        index = len(logits.shape) - 1

        logits = tf.reshape(logits, (-1, logits.shape[index]))
        labels = tf.reshape(labels, (-1, labels.shape[index]))
        logits = tf.boolean_mask(logits, tf.reduce_sum(labels, 1) > 0)
        labels = tf.boolean_mask(labels, tf.reduce_sum(labels, 1) > 0)

        if name == 'one-hot-accuracy':
            return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
        elif name == 'rmsp':
            return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(1 - labels, logits))))
        raise BaseException("Invalid accuracy function.")

    def get_one_hot_loss(logits, labels, name, cost_mask=None):
        index = len(labels.shape) - 1
        invalid_mask = tf.expand_dims(tf.cast(tf.reduce_sum(labels, index) > 0, dtype=tf.float32), -1)
        logits = tf.multiply(logits, invalid_mask)
        total_length = tf.reduce_sum(invalid_mask)

        if name == 'softmax-rmse':
            logits = tf.nn.softmax(logits)
            return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(labels, logits))) / total_length)
        elif name == 'weighted-sed-softmax-rmse':
            assert cost_mask is not None, "Specify a valid cost mask array."

            logits = tf.nn.softmax(logits)
            rmse = tf.reduce_mean(tf.square(tf.subtract(labels, logits)), index)
            rmse = tf.multiply(rmse, tf.reduce_sum(tf.multiply(cost_mask, labels), index))
            rmse = tf.multiply(rmse, 1 + tf.square(tf.cast(tf.argmax(labels, index) - tf.argmax(logits, index), dtype=tf.float32)))
            rmse = tf.divide(tf.reduce_sum(rmse), total_length)
            return tf.sqrt(rmse)
        elif name == 'softmax-cross-entropy':
            return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)) / total_length
        elif name == 'weighted-sed-softmax-cross-entropy':
            assert cost_mask is not None, "Specify a valid cost mask array."
            
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            cross_entropy = tf.multiply(cross_entropy, tf.reduce_sum(tf.multiply(cost_mask, labels), index))
            cross_entropy = tf.multiply(cross_entropy, 1 + tf.square(tf.cast(tf.argmax(labels, index) - tf.argmax(logits, index), dtype=tf.float32)))
            cross_entropy = tf.divide(tf.reduce_sum(cross_entropy), total_length)
            return cross_entropy
        raise BaseException("Invalid loss function.")

    def get_sdae_loss(logits, labels, name):
        if name == 'rmse':
            return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, logits))))
        elif name == 'sigmoid-cross-entropy':
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        raise BaseException("Invalid loss function.")

    def get_initializater(name):
        if name == 'uniform':
            return tf.random_uniform_initializer(-1, 1)
        elif name == 'xavier':
            return tf.contrib.layers.xavier_initializer()
        raise BaseException("Invalid initializer.")

    def get_optimizer(name, learning_rate):
        if name == 'gradient-descent':
            return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif name == 'adam':
            return tf.train.AdamOptimizer(learning_rate=learning_rate)
        raise BaseException("Invalid optimizer.")

    def get_learning_rate(name, learning_rate, step):
        if name == 'none':
            return learning_rate
        elif name == 'fraction':
            return learning_rate / (1 + step)
        elif name == 'exponential':
            return learning_rate * math.pow(0.99, float(step))
        raise BaseException("Invalid learning rate.")

    def classes_matching_matrix(Y):
        result = np.copy(Y)
        result[result == 0] = -1
        return np.transpose(result)

    def homogenize(X, Y, ratio_threshold=1): #TODO: add also class0 records?
        assert ratio_threshold > 0 and ratio_threshold <= 1, "Invalid ratio threshold."
        class_num = len(Y[0])
        class_occurrences = np.int32(np.sum(Y, 0))
        class_max_occurrence = np.int32(np.max(class_occurrences) * ratio_threshold)
        class_indexes = [np.where((np.argmax(Y,1) == i) & (np.sum(Y,1) > 0)) for i in range(class_num)]

        newX = []
        newY = []
        for i in range(len(class_indexes)):
            if(class_occurrences[i] >= class_max_occurrence):
                idx = range(class_occurrences[i])
            elif(class_occurrences[i] > 0):
                idx = np.random.choice(len(class_indexes[i][0]), class_max_occurrence)
            else:
                continue
            for j in idx:
                newX.append(X[class_indexes[i][0][j]])
                newY.append(Y[class_indexes[i][0][j]])

        return np.array(newX), np.array(newY)

    def get_batch(X, Y, size): 
        assert size > 0, "Size should positive"
        idx = np.random.choice(len(X), size, replace=False)
        return X[idx], Y[idx]
    
    def get_sequential_batch(X, Y, start, size):
        assert size > 0, "Size should positive"
        assert start >= 0, "Start should not be negative"   
        return X[start:start+size], Y[start:start+size]

    def get_rnn_batch(X, Y, lengths, size): 
        assert size > 0, "Size should positive"
        idx = np.random.choice(len(X), size, replace=False)
        return X[idx], Y[idx], lengths[idx]

    def get_rnn_sequential_batch(X, Y, lengths, start, size):
        assert size > 0, "Size should positive"
        assert start >= 0, "Start should not be negative"   
        return X[start:start+size], Y[start:start+size], lengths[start:start+size]

    def generate_flat_train_test(X, Y, training_fraction):
        indexes = np.random.rand(X.shape[0]) < training_fraction
        return [X[indexes], Y[indexes]], [X[~indexes], Y[~indexes]]

    def generate_rnn_train_test(X, Y, lengths, training_fraction):
        indexes = np.random.rand(X.shape[0]) < training_fraction
        return [X[indexes], Y[indexes], lengths[indexes]], [X[~indexes], Y[~indexes], lengths[~indexes]]

    def get_cost_mask(Y):
        axes = tuple(range(len(Y.shape) - 1))
        class_occurrences = np.int32(np.sum(Y, axes))
        class_max_occurrence = np.int32(np.max(class_occurrences))
        return np.array(class_max_occurrence / class_occurrences, dtype=np.float32)

    def rnn_shift_padding(X, Y, max_sequence_length): #TODO fix truncate case
        assert len(X) > 0, "Dataset should have at least one timeseries"
        assert len(X) == len(Y), "Input and classes should have the same length"
        assert max_sequence_length > 0, "Max sequence length should be positive" 
        dim = len(X)
        input_size = len(X[0][0])
        class_size = len(Y[0][0])
        newX = []
        newY = []
        newYincr = []
        sequence_length = []

        for index in range(len(Y)):
            start = 0
            end = 0

            if(np.max(Y[index]) > 0): #at least 1 valid class
                start = np.where(np.array([np.max(wave) for wave in Y[index]]) > 0)[0][0]
                end = np.where(np.array([np.max(wave) for wave in Y[index]]) > 0)[0][-1:][0]
            if(end - start > max_sequence_length):
                end = start + max_sequence_length
            length = end - start

            shifted_classes = np.concatenate((Y[index][1:], [np.zeros(class_size)]))
            classes = np.copy(Y[index])
            zeros = [[0] for i in range(len(classes))]
            classes_value = np.hstack((zeros, classes))
            classes_value = np.argmax(classes_value, 1)
            yesno = []

            last_found = 0
            for i in range(len(classes_value)):
                if classes_value[i] == 0:
                    yesno.append([0., 0.])
                elif last_found == 0:
                    yesno.append([0., 0.])
                    last_found = classes_value[i]
                elif classes_value[i] > last_found:
                    yesno.append([0., 1.])
                    last_found = classes_value[i]
                else:
                    yesno.append([1., 0.])
                    last_found = classes_value[i]

            yesno = np.concatenate((yesno[1:], [[0., 0.]]))

            waves_indexes = np.arange(start, end)
            if(length < max_sequence_length):
                tmpX = np.concatenate((X[index][waves_indexes], [np.zeros(input_size) for i in range(length, max_sequence_length)]))
                tmpY = np.concatenate((shifted_classes[waves_indexes], [np.zeros(class_size) for i in range(length, max_sequence_length)]))
                tmpYincr = np.concatenate((np.array(yesno[waves_indexes]), [[0., 0.] for i in range(length, max_sequence_length)]))
            else:
                tmpX = np.array(X[index][waves_indexes])
                tmpY = np.array(shifted_classes[waves_indexes])
                tmpYincr = np.array(yesno[waves_indexes])

            if length > 0:
                newX.append(tmpX)
                newY.append(tmpY)
                newYincr.append(tmpYincr)
                sequence_length.append(length)
        return np.array(newX), np.array(newY), np.array(newYincr), np.array(sequence_length)
    
    def add_noise(x, noise):
        shape = x.shape
        x = np.reshape(x, (-1, shape[-1]))
        result = []
        if noise == 'none':
            result = x
        elif noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            result = x + n
        elif 'mask' in noise:
            frac = float(noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), int(round(frac * len(i))), replace=False)
                i[n] = 0
            result = temp   
        result = np.reshape(result, shape)
        return result

    def noise_validator(noises):
        if not isinstance(noises, list):
            noises = [noises]
        for n in noises:
            try:
                if n in ['none', 'gaussian']:
                    return True
                elif n.split('-')[0] == 'mask' and float(n.split('-')[1]):
                    t = float(n.split('-')[1])
                    if t >= 0.0 and t <= 1.0:
                        return True
                    else:
                        return False
            except:
                return False
            pass
        return False
