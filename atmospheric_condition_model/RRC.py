import numpy as np
import pickle, glob, sys, os

from utils import *
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix

# Save model in path with name
def save_model(C_, path, name):
    with open(path.format(name), 'wb') as f:
        pickle.dump(C_, f)

def _polynomial_least_squares_fit(X_, Y, degree, l):
    X_ = PolynomialFeatures(int(degree)).fit_transform(X_)
    A_ = np.matmul(X_.T, X_) + l*np.identity(X_.shape[1])*X_.shape[0]
    return np.matmul(np.linalg.inv(A_), np.matmul(X_.T, Y_))

def _polynomial_least_squares_model(X_, w_, degree):
    X_ = PolynomialFeatures(degree).fit_transform(X_)
    return np.matmul(X_, w_)

def _classification(y_ts_, y_ts_hat_, y_lower = 0, y_upper = 2):
    # Find Classification Labels
    y_bounds_ = np.unique(y_ts_)
    y_lower, y_upper = y_bounds_[0], y_bounds_[-1]
    # Regression Discretization for classification
    y_ts_hat_ = np.around(y_ts_hat_)
    y_ts_hat_[y_ts_hat_ < y_lower] = y_lower
    y_ts_hat_[y_ts_hat_ > y_upper] = y_upper
    # Transform labels from float to integer
    return y_ts_hat_.astype(int)

# def _get_subsets(X_, y_, percentege = 0.75):
#     np.random.seed(0)
#     N    = X_.shape[0]
#     N_tr = int(N * percentege)
#     N_ts = N - N_tr
#     idx_ = np.random.permutation(N)
#     return X_[idx_[:N_tr], :], y_[idx_[:N_tr]], X_[idx_[-N_ts:], :], y_[idx_[-N_ts:]]

def _cross_validation(X_tr_, y_tr_, _degree, L = 10, K = 5):
    return

var_list_ = [[0, 1], [0, 1, 2, 3], [0, 1, 4], [0, 1, 4, 5], [0, 1, 4, 7]][]

# Input and output name
file_name = r''
path = '/users/terren/atmospheric_condition_model/software/{}'

# Index to the dataset
degree   = int(sys.argv[1])
var_idx_ = int(sys.argv[2])
var_     = var_list_[var_idx_]
# Display Experiment Configuration
print('>> Degree: {} Var. Set: {}'.format(degree, var_))


# # Number of polynomial order, Tikhonov Regularization, and K-Fold
# P, R, K = 6, 50, 10
# # Load up feature extraction results
# X_, Y_ = _load_file(path.format(file_name))[0]
# # Set it in matrix form
# XX_, YY_ = [], []
# for x_, y_ in zip(X_, Y_):
#     x_ = np.squeeze(np.concatenate(x_, axis = 1))
#     #x_ = np.concatenate(x_, axis = 1).T
#     x_ = x_[idx_set_, ...]
#     y_ = np.ones((x_.shape[0])) * y_
#     print(x_.shape, y_.shape)
#     # Make Autoregressive data
#     XX_.append(x_)
#     YY_.append(y_)
# # Form Dataset in matrix form
# X_, y_ = np.concatenate(XX_, axis = 0), np.concatenate(YY_, axis = 0)
# print(X_.shape, y_.shape)
# # Training and Test subsets
# X_tr_, y_tr_, X_ts_, y_ts_ = _get_subsets(X_, y_ )
# print(X_tr_.shape, y_tr_.shape, X_ts_.shape, y_ts_.shape)
#
# # timing start
# t_0 = datetime.now()
#
# # Variables Initialization
# p_0 = 0
# p_ = np.arange(p_0 + 1, P + 1 , dtype = int) # Polynomial Order
# E_ = np.zeros((P - p_0, R, K))
# #p_ = np.arange(1, P + 1 , dtype = int) # Polynomial Order
# r_ = np.logspace(-12, 12, R) # Regularization
# #E_ = np.zeros((P, R, K))
# # K-fold Cross-Validation Variables
# N    = X_tr_.shape[0]
# idx_ = np.random.permutation(N)
# n_samples_fold = N//K
# # Loop over Polynomal orders
# for p, i in zip(p_, range(r_.shape[0])):
#     # Loop over regularization values
#     for l, ii in zip(r_, range(r_.shape[0])):
#         # Loop over K-Fold Cross-Calidation
#         for k in range(K):
#             # Cross-Validation Index
#             idx_test  = idx_[k*n_samples_fold:(k + 1)*n_samples_fold]
#             idx_train = np.setxor1d(idx_, idx_[k*n_samples_fold:(k + 1)*n_samples_fold])
#             # Model Fit
#             w_ = _polynomial_least_squares_fit(X_tr_[idx_train, :], y_tr_[idx_train], p, l)
#             # Model Prediction
#             y_val_hat_, y_val_ = _polynomial_least_squares_model(X_tr_[idx_test, :], w_, p), y_tr_[idx_test]
#             # Model Score
#             y_ts_hat_    = _classification(y_val_, y_val_hat_)
#             E_[i, ii, k] = accuracy_score(y_val_, y_ts_hat_)
#
# # Validation Results
# e_ = np.mean(E_, axis = 2)
# print(p_)
# print(e_)
# i, ii = np.where(e_ == np.max(e_))
# print(p_[i], r_[ii], np.max(e_))
#
# # timing stop
# t_1 = datetime.now()
# print(t_1 - t_0)
#
# # Fit Model to save
# n, l = p_[i][0], r_[ii][0]
# w_ = _polynomial_least_squares_fit(X_tr_, y_tr_, n, l)
# # Replicate Train
# y_tr_hat_ = _polynomial_least_squares_model(X_tr_, w_, n)
# y_tr_hat_ = _classification(y_tr_, y_tr_hat_)
# # Scores in train
# e, m = accuracy_score(y_tr_, y_tr_hat_), confusion_matrix(y_tr_, y_tr_hat_)
# print(e)
# print(m)
# # Replicate Test
# y_ts_hat_ = _polynomial_least_squares_model(X_ts_, w_, n)
# y_ts_hat_ = _classification(y_ts_, y_ts_hat_)
# # Scores in Test
# e, m = accuracy_score(y_ts_, y_ts_hat_), confusion_matrix(y_ts_, y_ts_hat_)
# print(e)
# print(m)

# Save Model
#print(w_.shape, n, l)
#save_model(C_ = [w_, n, l], path = path, name = save_name)
