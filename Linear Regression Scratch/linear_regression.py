import numpy as np
import logging

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

class LinearRegression():
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.biases = None
        self.history = {"loss": []} # for keeping track of the loss in every iteration
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            print(f"{i+1} iteration, weight: {self.weights} ")
            self.bias -= self.learning_rate * db
            print(f"{i+1} iteration, bias: {self.bias} ")
            
            loss = np.mean((y_pred - y) ** 2)
            self.history['loss'].append(loss)
            print(f"{i+1} iteration, loss: {loss}")
        
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        
    

        