import numpy as np
import matplotlib.pyplot as plt


data = np.load("c_arrays.npz")
for key in data.keys():
    normerror = [np.linalg.norm(x) for x in data[key]]
    normerror -= normerror[-1]
    # normerror /= np.max(np.abs(normerror))
    plt.plot(np.abs(normerror)[:-1], label=key)

plt.yscale('log')
plt.legend()
plt.show()
