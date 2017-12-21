import numpy as np
import tensorflow as tf
from sklearn import linear_model as lm
from utils import Utils as utils

class Svm:

    def __init__(self, printer, output_size, cost_mask, loss='hinge', penalty='l2', alpha=0.0001, batch_size=1000):
        self.printer = printer
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
        
        true_positives = 0
        for i in range(len(outputs)):
            self.confusion_matrix[Y[i]][outputs[i]] += 1
            if Y[i] == outputs[i]:
                true_positives += 1
        
        results = {}
        real = np.sum(self.confusion_matrix, 1)
        predicted = np.sum(self.confusion_matrix, 0)
        total = np.sum(self.confusion_matrix)
        
        sum_classifier = 0
        sum_prob_classifier = 0
        accuracy = 0
        for i in range(len(self.confusion_matrix)):
            sum_classifier += self.confusion_matrix[i][i]
            sum_prob_classifier += real[i] * predicted[i] / total
        results['accuracy'] = sum_classifier / total
        results['k-statistic'] = (sum_classifier - sum_prob_classifier) / (total - sum_prob_classifier)
        #results['confusion-matrix'] = self.confusion_matrix
        
        for m in results:
            self.printer.print('\t {0}: {1}'.format(m, results[m]))
        return results
