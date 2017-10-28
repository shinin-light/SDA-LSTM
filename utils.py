import numpy as np

class DatasetUtils:

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
            else:
                idx = np.random.choice(len(class_indexes[i][0]), class_max_occurrence)
            for j in idx:
                newX.append(X[class_indexes[i][0][j]])
                newY.append(Y[class_indexes[i][0][j]])

        return np.array(newX, dtype=np.float32), np.array(newY, dtype=np.float32)
    
class NeuralNetworkUtils:

    def get_mask(Y):
        class_occurrences = np.int32(np.sum(Y, 0))
        class_max_occurrence = np.int32(np.max(class_occurrences))
        return class_max_occurrence / class_occurrences

