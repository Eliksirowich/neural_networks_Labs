import pandas as pd
import numpy as np
from neural import Perceptron


df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values


inputSize = X.shape[1]  
hiddenSizes1 = 10  
hiddenSizes2 = 10
outputSize = 1 if len(y.shape) else y.shape[1] 


NN = Perceptron(inputSize, hiddenSizes1, hiddenSizes2, outputSize)

NN.train(X, y, n_iter=5, eta = 0.01)

y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[:, [0, 2]].values
outs = []
for xi in X:
    out, h1, h2 = NN.predict(xi)
    outs.append(out)

outs = np.array(outs)

sum(out-y.reshape(-1, 1))
