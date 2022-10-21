import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
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

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # Set up marker deneration and color map
    markers = ('o', 's', '^', 'v', ',')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cian')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Plot the decision surface
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0],
            y=X[y==cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f'Class{cl}',
            edgecolor='black')

X_combined_std = np.vstack((X_train_STD, X_test_STD))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))

plt.xlabel('Petel Length [Standardised]')
plt.ylabel('Petal width [Standardised]')
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()