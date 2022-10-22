import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1 / (1.0 +np.exp(-z))

z = np.arange(-7, 7, 0.1)
sigma_z = sigmoid(z)
plt.plot(z, sigma_z)
plt.axvline(0.0, color='k')
plt.ylim=(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\sigma (z)$')
plt.yticks([0.0, 0.5, 1])
ax = plt.gca()
plt.tight_layout()
plt.show()