import numpy as np
import glob
import os

for f in glob.glob('runs/run-*'):
    print (f)
    x = np.load(os.path.join(f, 'results.npy'))
    print (x.shape, np.mean(x[-100:]), np.std(x[-100:]))