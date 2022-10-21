import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron 
import numpy as np

# Loading Data
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target


# Splitting Data into Train and Test 
print('Class labels:', np.unique(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
    random_state=1, stratify=y)

# Standardazind Data
sc = StandardScaler()
sc.fit(X_train)
X_train_STD = sc.transform(X_train)
X_test_STD = sc.transform(X_test)

# Predicting
ppn = Perceptron(eta0=0.01, random_state=1)
ppn.fit(X_train_STD, y_train)
y_pred = ppn.predict(X_test_STD)
print('Misclassified examples: %d' % (y_test !=y_pred).sum())