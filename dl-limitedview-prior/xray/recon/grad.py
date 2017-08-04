
import numpy as np
from scipy.sparse import csr_matrix
import os.path

def read_system_matrix(dirname):
    NX, NY = 367*120, 256*256 
    irows = np.fromfile(os.path.join(dirname, 'irows.dat'), dtype=np.float32)
    icols = np.fromfile(os.path.join(dirname, 'icols.dat'), dtype=np.float32)
    vals = np.fromfile(os.path.join(dirname, 'vals.dat'), dtype=np.float32)
    return csr_matrix( (vals, (irows, icols)), shape=(NX, NY) )

def gradient(H, x):
    return H.transpose().dot(H.dot(x))

