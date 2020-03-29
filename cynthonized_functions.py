import numpy as np
from scipy.stats import entropy
from cython cimport parallel
from libc.math cimport log
import cython
from cython.view cimport array as cvarray
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.



def score_entropy(prediction_distribution):
  results = np.zeros(shape = prediction_distribution.shape, dtype=np.dtype("d"))
  score_entropy_parallel(prediction_distribution, results)
  results = np.asarray(results)
  entropy_images = np.sum(np.sum(results, axis = 2), axis = 1)
  return entropy_images



@cython.boundscheck(False)
@cython.wraparound(False)
cdef int score_entropy_parallel(double[:,:,:] pred_dist, double[:,:,:] results) nogil:
    # Py_ssize_t is the proper C type for Python array indices.
    cdef Py_ssize_t i
    cdef int n = pred_dist.shape[0], m = pred_dist.shape[1], o = pred_dist.shape[2]
    for i in xrange(n):
        for j in xrange(m):
          for k in xrange(o):
            if pred_dist[i, j, k] != 0:
              results[i,j,k] = -1*pred_dist[i, j, k]*log(pred_dist[i, j, k]) - (1-pred_dist[i, j, k])*log(1-pred_dist[i, j, k])
            else:
              results[i,j,k] = 0
    return 0
