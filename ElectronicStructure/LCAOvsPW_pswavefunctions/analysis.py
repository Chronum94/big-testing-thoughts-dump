import numpy as np
import matplotlib.pyplot as plt


pspw = np.load("mo_pspw.npz")['rho']
pslcao = np.load("mo_pslcao.npz")['rho']

# fig, ax = plt.subplots(1, 2)

# map1 = ax[0].imshow(np.log(pspw[30]))
# plt.colorbar(mappable=map1, ax=ax[0])
# map2 = ax[1].imshow(np.log(pslcao[30]))
# plt.colorbar(mappable=map2, ax=ax[1])
# plt.show()

fig, ax = plt.subplots(1, 1)

data = pslcao[60] - pspw[60]
plt.imshow(np.log(np.abs(data)))
plt.colorbar()
plt.show()
