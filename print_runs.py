import numpy as np
import glob
import os

for f in glob.glob('runs/*'):
    print (f)
    for f2 in glob.glob(os.path.join(f, '*.npy')):
        x = np.load(f2)
        print (' ', os.path.basename(f2) + ':', x.shape, np.mean(x[-100:-1]), np.std(x[-100:-1]))
