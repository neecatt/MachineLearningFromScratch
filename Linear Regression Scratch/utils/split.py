import numpy as np


def train_test_split(X, y, test_size=None, random_state=None, shuffle=None):
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    if len(X) != len(y):
        raise ValueError("Feature and target matrices should have the same length")
    
    if isinstance(test_size, float):
        test_size = int(len(X) * test_size)
    elif isinstance(test_size, int):
        if test_size >= len(X):
            raise ValueError("Test size should be smaller than dataset size")
    else:
        raise ValueError("Test size should be a float or an integer")
    
    indices = np.arange(len(X))
    print(indices)
    
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state) # We use random state to ensure that we get the same random shuffle every time
        np.random.shuffle(indices)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
            
        
#TODO: fix indexing error in this code