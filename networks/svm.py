import numpy as np
import tensorflow as tf
from sklearn import svm
from utils import Utils as utils

class Svm:

    def __init__(self):
        self.classifier = svm.SVC(gamma=0.001)

    def train(self, X, Y):
        Y = np.argmax(Y, 1)
        self.classifier.fit(X, Y)

    def test(self, X, Y):
        output_size = len(Y[0])
        Y = np.argmax(Y, 1)
        outputs = self.classifier.predict(X)
        
        counters = [[0 for i in range(output_size)] for j in range(output_size)]
        for i in range(len(outputs)):
            counters[Y[i]][outputs[i]] += 1

        [print("class {0}, accuracy = {1:.2f}, values =".format(i+1, counters[i][i] / np.sum(counters[i])), counters[i]) for i in range(len(counters))]
