import numpy as np

def z_score_normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis =0)
    
    std[std == 0] = 1 # To prevent zero division error
    
    X_standardized = (X - mean) / std
    return X_standardized, mean, std # Returning mean and std as well, so we can use the same mean and std for training and test data
    