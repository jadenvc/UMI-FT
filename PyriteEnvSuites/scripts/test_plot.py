import matplotlib.pyplot as plt
import numpy as np

filename = "/home/yifanhou/git/force_record1_k0.npy"

data = np.load(filename)
print(data.shape)

mag = np.linalg.norm(data[:, :], axis=1)

plt.plot(mag)
plt.show()
