import numpy as np
import matplotlib.pyplot as plt

"""
Simple demo of the fill function
"""

x = np.linspace(-0.2,1)
y = np.sin(4 * np.pi * x) * np.exp(-5 * x)

plt.fill(x, y, 'r')
plt.grid(True)
plt.show()
