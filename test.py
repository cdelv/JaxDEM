import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def histogram(val, use_log_bins=False):
    n = int(np.sqrt(val.size))
    if use_log_bins:
        bins = np.logspace(np.min(val[val > 0]), np.max(val), n)
    else:
        bins = np.linspace(np.min(val), np.max(val), n)
    p, edge = np.histogram(val, bins, density=True)
    return p, (edge[1:] + edge[:-1]) / 2

root = 'delete-this-data'
xs, ys, cs = [], [], []
for fname in os.listdir(root):
    data = np.load(os.path.join(root, fname))
    p, x = histogram(data['mu'].ravel())
    xs.append(x)
    ys.append(p)
    cs.append(data['tracer_radius'] / data['asperity_radius'])

cmap = plt.cm.viridis
norm = LogNorm(min(cs), max(cs))

for x, y, c in zip(xs, ys, cs):
    plt.plot(x, y, c=cmap(norm(c)))

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array(cs)
cbar = plt.colorbar(sm, ax=plt.gca(), label=r'$\sigma_t / \sigma_v$')

plt.yscale('log')
plt.xscale('log')
plt.savefig('test.png')