import os
import numpy as np
import matplotlib.pyplot as plt

def histogram(val, use_log_bins=False):
    n = int(np.sqrt(val.size))
    if use_log_bins:
        bins = np.logspace(np.min(val[val > 0]), np.max(val), n)
    else:
        bins = np.linspace(np.min(val), np.max(val), n)
    p, edge = np.histogram(val, bins, density=True)
    return p, (edge[1:] + edge[:-1]) / 2

root = 'delete-this-data'
for fname in os.listdir(root):
    data = np.load(os.path.join(root, fname))
    p, x = histogram(data['mu'].ravel())
    plt.plot(x, p)
plt.yscale('log')
plt.xscale('log')
plt.savefig('test.png')