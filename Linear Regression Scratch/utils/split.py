import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    if len(X) != len(y):
        raise ValueError("Feature and target matrices should have the same length")
    
    if test_size is None:
        test_size = 0.2 
        
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("Float test size should be between 0 and 1")
        test_size = int(len(X) * test_size)
    elif isinstance(test_size, int):
        if test_size >= len(X):
            raise ValueError("Test size should be smaller than dataset size")
    else:
        raise ValueError("Test size should be a float or an integer")
    
    indices = np.arange(len(X))
    
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state) # We use random state to ensure that we get the same random shuffle every time
        np.random.shuffle(indices)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
