import numpy as np

""""
method for normalization
"""

def z_score(series):
   
    result = (series - series.mean()) / series.std(ddof=0)
    return result

def point_relative_normalization(series):
   
    result = (series / series.values[0]) - 1
    return result

def maximum_and_minimum_normalization(series, boundary=(0, 1)):
    
    range_min, range_max = boundary
    standard_deviation = (series - series.min(axis=0)) / (series.max(axis=0) - series.min(axis=0))
    result = standard_deviation * (range_max - range_min) + range_min

    return result

"""
method for create model
"""

def create_groups(data, group_size=4):
   
    samples = list()
    for i in range(0, len(data), group_size):
        sample = list(data[i:i + group_size])
        if len(sample) == group_size:
            samples.append(np.array(sample).reshape(1, group_size).tolist())

    A = np.array(samples)
    return A.reshape(1, A.shape[0], group_size)

def split_lstm_input(groups):
   
    X = groups[0:,:-1].reshape(1, groups.shape[1] - 1, groups.shape[2])
    Y = groups[0:,-1:][0]

    return X, Y

"""
method for evaluate model
"""

def mape(A, B):
    
    return np.mean(np.abs((A - B) / A)) * 100

def rmse(A, B):
   
    return np.sqrt(np.square(np.subtract(A, B)).mean())
