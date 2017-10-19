import numpy as np


def get_batch(X, X_, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a], X_[a]


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
