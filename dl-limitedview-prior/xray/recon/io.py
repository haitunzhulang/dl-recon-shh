
import os
import os.path
import numpy as np

# CONSTANTS
NX = 256 # Number of pixels along the x-direction of the image
NY = 256 # Number of pixels along the y-direction of the image
NRAYSv1 = 367
NRAYSv2 = 256
NRAYSv3 = 256
NVIEWS = 120 
NLABELS = 2
NPIXELS = NX * NY # Total number of pixels in the image
NDATAv1 = NRAYSv1 * NVIEWS
NDATAv2 = NRAYSv2 * NVIEWS

H_FILENAME_PREFIXv1 = 'H120v2_'
H_FILENAME_PREFIXv2 = 'H120v3_'

def get_max_nsamples(dirname):
    """Calculate the number of samples in a given directory

    This function assumes the samples have the form:
        img#.dat or recon#.dat

    Arguments:
      dirname - directory name where the samples are located
    Returns:
      A scalar indicating the total number of samples in the 
      directory.
    """
    nfiles = 0
    for fn in os.listdir(dirname):
       if fn.startswith('img') or fn.startswith('recon'):
           nfiles += 1 
    return nfiles

def read_system_matrix(dirname, outtype='tf', version=2,theta=None):
    """Read sparse system matrix from disk

    Reads in a pre-calculated system matrix from disk that models
    parallel-beam X-ray CT propagation. For more information on how
    the system matrix was calculated see the GitLab project:
        radon.seas.wustl.edu/xray/xct-parallelbeam-matlab
    This system matrix corresponds to an input of a 256 x 256 image
    and an output of 120 views with NRAYS detectors at each view. The
    value of NRAYS will depend on ther version of the system matrix.
    For version 1, NRAYS = 367. For subsequent versions, NRAYS = 256.

    Arguments:
      dirname - directory name where files are located
      outtype - either 'tf' or 'scipy'; in the first case, the output
          will have type tf.SparseTensor, in the second, the output 
          will have type scipy.sparse.csr_matrix (Default: 'tf')
      version - integer specifying which version of the system matrix
          to read in. (Default: 2)
    Returns:
      Either a tf.SparseTensor or scipy.sparse.csr_matrix with 
      dimensions NDATA x NPIXELS. For version 1, NDATA = 367*120
      while for version 2, NDATA = 256*120. For both versions,
      NPIXELS = 256*256.
    """
    if theta is None:
        if version == 1:
            fn_prefix = H_FILENAME_PREFIXv1
            ndata = NDATAv1
        else:
            fn_prefix = H_FILENAME_PREFIXv2
            ndata = NDATAv2
    else:
        fn_prefix='H' + str(theta) + 'v3_'
        ndata = NRAYSv2 * theta
        
    
    irows = np.fromfile(os.path.join(dirname, '%sirows.dat' % \
        fn_prefix), dtype=np.float32)
    icols = np.fromfile(os.path.join(dirname, '%sicols.dat' % \
        fn_prefix), dtype=np.float32)
    vals = np.fromfile(os.path.join(dirname, '%svals.dat' % \
        fn_prefix), dtype=np.float32)
    if outtype == 'tf':
        from tensorflow import SparseTensor
        indices = np.zeros( (int(vals.shape[0]), 2), dtype=np.int64 )
        indices[:,0] = irows.astype(np.int64) - 1
        indices[:,1] = icols.astype(np.int64) - 1
        return SparseTensor(indices, vals, shape=(ndata, NPIXELS))
    from scipy.sparse import csr_matrix
    return csr_matrix( (vals, (irows-1, icols-1)), shape=(ndata, NPIXELS) )

def read_float_data(dirname, fn_prefix, ind, version=2, theta=0):
    """Reads binary single-precision floating-point data from disk

    Arguments:
      dirname - directory name where the data is located
      fn_prefix - filename prefix, assumes the filename is of the
          form: <FN_PREFIX><IND>.dat
      ind - the index of the sample

    Returns:
      A 1D float32 numpy array containing the data read from disk
    """
#     print('trying something 2')
    if version==3:
        if theta > 0:
            filename = os.path.join(dirname, '%s%d_%d.dat' % (fn_prefix, theta, ind))
        else: 
            filename = os.path.join(dirname, '%s_%d.dat' % (fn_prefix, ind))
    else:
        filename = os.path.join(dirname, '%s%d.dat' % (fn_prefix, ind))
    
#     print('filename: ' + filename)
    return np.fromfile(filename, dtype=np.float32)

def read_meas_data(dirname, ind, theta=0, version=2):
    if version==3:
        return read_float_data(dirname, 'measdata_nviews', ind, version, theta)
        
    return read_float_data(dirname, 'measdata', ind, version, theta)

def read_true_image(dirname, ind, version=2):
    return read_float_data(dirname, 'img', ind, version)

def read_recon_image(dirname, ind, theta=0, version=2):
    if version ==3:
        return read_float_data(dirname, 'recon_nviews', ind, version, theta)
    
    return read_float_data(dirname, 'recon', ind, version, theta)

def read_images(dirname, nsamples, one_hot=False, verbose=False):
    PRINT_RATE = 100
    if get_max_nsamples(dirname) < nsamples:
        raise ValueError('Not enough samples available.')
    if nsamples % 2 == 1:
        raise ValueError('Number of samples must be even.')

    images = np.zeros( (nsamples, NPIXELS), dtype=np.float32)
    labels_dense = np.zeros(nsamples, dtype=np.float32)

    for i in range(nsamples//2): # integer division
        if verbose and i % PRINT_RATE == 0:
            print('Reading image', i)
        images[2*i,:] = read_true_image(dirname, i)
        images[2*i+1,:] = read_recon_image(dirname, i)
        labels_dense[2*i] = 0
        labels_dense[2*i+1] = 1
    
    # Scale images to undo scaling by MNIST DataSet class
    images = np.multiply(images, 255.0)

    labels = labels_dense
    if one_hot: # only works if NLABELS = 2
        labels = np.zeros( (nsamples, NLABELS), dtype=np.float32)
        labels[:,0] = [label for label in labels_dense]
        labels[:,1] = [1-label for label in labels_dense]
    
    return images, labels




