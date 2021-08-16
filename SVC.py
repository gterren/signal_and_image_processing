import numpy as np
import pickle, glob, sys, os

from utils import *
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.svm import LinearSVC

# Save model in path with name
def save_model(C_, path, name):
    with open(path.format(name), 'wb') as f:
        pickle.dump(C_, f)

# def _get_subsets(X_, y_, percentege = 0.75):
#     np.random.seed(0)
#     N    = X_.shape[0]
#     N_tr = int(N * percentege)
#     N_ts = N - N_tr
#     idx_ = np.random.permutation(N)
#     return X_[idx_[:N_tr], :], y_[idx_[:N_tr]], X_[idx_[-N_ts:], :], y_[idx_[-N_ts:]]
#
# # ... Regularization
# def _reg(x_, i_var):
#     if i_var == 10:
#         idx_ = x_[..., i_var] > 10
#         x_[idx_, i_var] = 0.
#     return x_
#
# def _get_stats(x_, vr_idx_):
#     x_mu_, x_s2_ = [], []
#     for i in vr_idx_:
#         # Regularization ...
#         x_mu_.append(np.mean(x_[..., i]))
#         x_s2_.append(np.std(x_[..., i]))
#     x_mu_ = np.array(x_mu_)
#     x_s2_ = np.array(x_s2_)
#     return np.concatenate((x_mu_, x_s2_), axis = 0)[np.newaxis, :]
#
# def _get_data(X_, Y_, vr_idx_):
#     xx_ = np.empty((0, 2*len(vr_idx_)))
#     yy_ = np.empty((0))
#     # Set it in matrix form
#     XX_, YY_ = [], []
#     for x_, y_ in zip(X_, Y_):
#         yy_ = np.concatenate((yy_, np.ones((len(x_))) * y_), axis = 0)
#         for i in range(len(x_)):
#             xxx_ = _get_stats(x_[i], vr_idx_)
#             xx_  = np.concatenate((xx_, xxx_), axis = 0)
#     return xx_, yy_

def _polynomial(X_, degree):
    return PolynomialFeatures(degree).fit_transform(X_)

def _cross_validation(X_tr_, y_tr_, degree, C, K):
    # Complexity and K-Fold
    E_ = np.zeros((C, K))
    c_ = np.logspace(-3, 3, C) # Complexity
    # K-fold Cross-Validation Variables
    N = X_tr_.shape[0]
    idx_ = np.random.permutation(N)
    n_samples_fold = N//K
    # Loop over Complexity values
    for c, i in zip(c_, range(c_.shape[0])):
        # Loop over K-Fold Cross-Calidation
        for k in range(K):
            # Cross-Validation Index
            idx_val_ = idx_[k*n_samples_fold:(k + 1)*n_samples_fold]
            idx_tr_  = np.setxor1d(idx_, idx_[k*n_samples_fold:(k + 1)*n_samples_fold])
            # SVM Fit
            _SVM = LinearSVC(C = c, multi_class = 'ovr', dual = False, fit_intercept = True)
            x_tr_ = _polynomial(X_tr_[idx_tr_, :], degree)
            _SVM.fit(x_tr_, y_tr_[idx_tr_])
            # SVM Prediction
            x_val_ = _polynomial(X_tr_[idx_val_, :], degree)
            y_val_hat_ = _SVM.predict(x_val_)
            y_val_ = y_tr_[idx_val_]
            # SVM Score
            E_[i, k] = accuracy_score(y_val_, y_val_hat_)
    # Validation Results
    e_ = np.mean(E_, axis = 1)
    i = np.where(e_ == np.max(e_))
    print(c_[i], np.max(e_))
    return c_[i]

def _train(X_tr_, y_tr_, _degree, c):
    # Define Model to save
    _SVM = LinearSVC(C = c, multi_class = 'ovr', dual = False, fit_intercept = True)
    # Fit Model
    x_tr_ = _polynomial(X_tr_, _degree)
    _SVM.fit(x_tr_, y_tr_)
    # Replicate Train
    y_tr_hat_ = _SVM.predict(x_tr_)
    # Scores in train
    e, m = accuracy_score(y_tr_, y_tr_hat_), confusion_matrix(y_tr_, y_tr_hat_)
    print(e)
    print(m)
    return _SVM

def _test(X_ts_, y_ts_, _degree, _SVM):
    # Replicate Test
    x_ts_ = _polynomial(X_ts_, _degree)
    y_ts_hat_ = _SVM.predict(x_ts_)
    # Scores in Test
    e, m = accuracy_score(y_ts_, y_ts_hat_), confusion_matrix(y_ts_, y_ts_hat_)
    print(e)
    print(m)

def _get_subsets(X_, y_, percentege = 0.75):
    np.random.seed(0)
    N    = X_.shape[0]
    N_tr = int(N * percentege)
    N_ts = N - N_tr
    idx_ = np.random.permutation(N)
    return X_[idx_[:N_tr], :], y_[idx_[:N_tr]], X_[idx_[-N_ts:], :], y_[idx_[-N_ts:]]

def _get_data(path):
    X_, Y_ = [], []
    for file in os.listdir(path):
        print(file)
        y_, x_ = _load_file(r'{}/{}'.format(path, file))[0]
        X_.append(x_)
        Y_.append(y_)

    return X_, Y_

var_list_ = [[0, 1], [0, 1, 2, 3], [0, 1, 4], [0, 1, 4, 5], [0, 1, 4, 7]]

# Input and output name

# Index to the dataset
degree   = int(sys.argv[1])
var_idx_ = int(sys.argv[2])
var_     = var_list_[var_idx_]
# Display Experiment Configuration
print('>> Degree: {} Var. Set: {}'.format(degree, var_))

X_, Y_ = _get_data(path = r'/users/terren/atmospheric_condition_model/data')
#
# # Load up feature extraction results
# X_, Y_ = _load_file(path.format(file_name))[0]
#
# X_, Y_ = _get_data(X_, Y_, vr_idx_)
#
# X_tr_, y_tr_, X_ts_, y_ts_ = _get_subsets(X_, Y_)
# print(X_tr_.shape, y_tr_.shape, X_ts_.shape, y_ts_.shape)
#
# # Complexity Parameters Cross-Validation
# c = _cross_validation(X_tr_, y_tr_, _degree, C = 10, K = 5)
# # Model Training
# _SVM = _train(X_tr_, y_tr_, _degree, c)
# # Display on Screen Testing results
# _test(X_ts_, y_ts_, _degree, _SVM)
# # Save Model
# print(_SVM, _degree)
#_save_model(C_ = [_SVM, p, c], path = path, name = save_name)
