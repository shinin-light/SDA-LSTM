import numpy as np
import tensorflow as tf
from sklearn import linear_model as lm
from utils import Utils as utils

class Svm:

    def __init__(self, output_size, cost_mask, loss='hinge', penalty='l2', alpha=0.0001, batch_size=1000):
        self.cost_mask = dict(zip(range(output_size), cost_mask))
        self.output_size = output_size
        self.batch_size = batch_size
        self.classifier = lm.SGDClassifier(max_iter=100, class_weight=self.cost_mask, loss=loss, penalty=penalty, alpha=alpha)
        
    def train(self, X, Y):
        idx = np.where(np.sum(Y, 1) > 0)[0]
        X = X[idx]
        Y = Y[idx]
        Y = np.argmax(Y, 1)
        
        batches = int(len(X) / self.batch_size)
        for i in range(batches):
            batch_x, batch_y = utils.get_batch(X, Y, self.batch_size)
            self.classifier.partial_fit(batch_x, batch_y, np.array(range(self.output_size)))

    def test(self, X, Y):
        output_size = len(Y[0])
        idx = np.where(np.sum(Y, 1) > 0)[0]
        X = X[idx]
        Y = Y[idx]
        Y = np.argmax(Y, 1)
        outputs = self.classifier.predict(X)
        
        self.confusion_matrix = [[0 for i in range(output_size)] for j in range(output_size)]
        for i in range(len(outputs)):
            self.confusion_matrix[Y[i]][outputs[i]] += 1

        [print("class {0}, accuracy = {1:.2f}, values =".format(i+1, self.confusion_matrix[i][i] / np.sum(self.confusion_matrix[i])), self.confusion_matrix[i]) for i in range(len(self.confusion_matrix))]
        return self.confusion_matrix
