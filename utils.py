import numpy as np

class Utils:

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

    def get_batch(X, X_, size):
        a = np.random.choice(len(X), size, replace=False)
        return X[a], X_[a]
    
    def generate_sdae_train_test(X, training_fraction):
        indexes = np.random.rand(X.shape[0]) < training_fraction
        return X[indexes], X[~indexes]

    def generate_classifier_train_test(X, Y, training_fraction):
        indexes = np.random.rand(X.shape[0]) < training_fraction
        return [X[indexes], Y[indexes]], [X[~indexes], Y[~indexes]]

    def generate_rnn_train_test(X, Y, lengths, training_fraction):
        indexes = np.random.rand(X.shape[0]) < training_fraction
        return [X[indexes], Y[indexes], lengths[indexes]], [X[~indexes], Y[~indexes], lengths[~indexes]]

    def get_cost_mask(Y):
        axes = tuple(range(len(Y.shape) - 1))
        class_occurrences = np.int32(np.sum(Y, axes))
        class_max_occurrence = np.int32(np.max(class_occurrences))
        return np.array(class_max_occurrence / class_occurrences)

    def rnn_shift_padding(X, X_, max_sequence_length): #TODO fix truncate case
        assert len(X) > 0, "Dataset should have at least one timeseries"
        assert len(X) == len(X_), "Input and classes should have the same length"
        assert max_sequence_length > 0, "Max sequence length should be positive" 
        dim = len(X)
        input_size = len(X[0][0])
        class_size = len(X_[0][0])
        newX = []
        newX_ = []
        sequence_length = []

        for index in range(len(X_)):
            start = 0
            end = 0

            if(np.max(X_[index]) > 0): #at least 1 valid class
                start = np.where(np.array([np.max(wave) for wave in X_[index]]) > 0)[0][0]
                end = np.where(np.array([np.max(wave) for wave in X_[index]]) > 0)[0][-1:][0]
            if(end > max_sequence_length):
                end = max_sequence_length
            length = end - start

            shifted_classes = np.concatenate((X_[index][1:], [np.zeros(class_size)]))
            waves_indexes = np.arange(start, end)
            if(length < max_sequence_length):
                tmpX = np.concatenate((X[index][waves_indexes], [np.zeros(input_size) for i in range(length, max_sequence_length)]))
                tmpX_ = np.concatenate((shifted_classes[waves_indexes], [np.zeros(class_size) for i in range(length, max_sequence_length)]))
            else:
                tmpX = np.array(X[index][waves_indexes])
                tmpX_ = np.array(shifted_classes[waves_indexes])
            
            if length > 0:
                newX.append(tmpX)
                newX_.append(tmpX_)
                sequence_length.append(length)
        return np.array(newX), np.array(newX_), np.array(sequence_length)
        
    #SDAE
    def noise_validator(noise, allowed_noises):
        '''Validates the noise provided'''
        for n in noise:
            try:
                if n in allowed_noises:
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
