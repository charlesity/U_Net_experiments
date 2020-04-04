import numpy as np
from scipy.stats import entropy
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon



def score_jensen(standard_prediction, average_prediction):
    results = np.zeros(shape = standard_prediction.shape[:-1], dtype=np.dtype("d"))
    score_jensen_parallel(standard_prediction, average_prediction, results)
    results = results.reshape(results.shape[0], results.shape[1]*results.shape[2])
    kl = results.sum(axis = 1)
    return kl



def score_kl(standard_prediction, average_prediction):
    results = np.zeros(shape = standard_prediction.shape[:-1], dtype=np.dtype("d"))
    score_kl_parallel(standard_prediction, average_prediction, results)
    results = results.reshape(results.shape[0], results.shape[1]*results.shape[2])
    kl = results.sum(axis = 1)
    return kl


def score_entropy(prediction_distribution):
  results = np.zeros(shape = prediction_distribution.shape[:-1], dtype=np.dtype("d"))
  score_entropy_parallel(prediction_distribution, results)
  results = results.reshape(results.shape[0], results.shape[1]*results.shape[2])
  ent = results.sum(axis = 1)
  return ent

def score_entropy_parallel(double[:,:,:,:] pred_dist, double[:,:,:] results):
    cdef Py_ssize_t img, i, j
    for img in range(pred_dist.shape[0]):
        for i in range(pred_dist.shape[1]):
          for j in range(pred_dist.shape[2]):
              results[img,i,j] = entropy(pred_dist[img,i,j,:])

def score_kl_parallel(double[:,:,:,:] stand_dist, double[:,:,:,:] avg_dist, double[:,:,:] results):
    cdef Py_ssize_t img, i, j
    for img in range(stand_dist.shape[0]):
        for i in range(stand_dist.shape[1]):
          for j in range(stand_dist.shape[2]):
              results[img,i,j] = kl_div(stand_dist[img,i,j,:] ,avg_dist[img,i,j,:]).sum()

def score_jensen_parallel(double[:,:,:,:] stand_dist, double[:,:,:,:] avg_dist, double[:,:,:] results):
    cdef Py_ssize_t img, i, j
    for img in range(stand_dist.shape[0]):
        for i in range(stand_dist.shape[1]):
          for j in range(stand_dist.shape[2]):
              results[img,i,j] = jensenshannon(stand_dist[img,i,j,:] ,avg_dist[img,i,j,:]).sum()
