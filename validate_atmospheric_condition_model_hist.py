import numpy as np

import pickle, glob, sys, os

from utils import *
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix

from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.stats import multivariate_normal, beta, gamma, norm, skewnorm, skew, kurtosis, t

from sklearn.svm import LinearSVC
from sklearn import mixture

# Save model in path with name
def save_model(C_, path, name):
    with open(path.format(name), 'wb') as f:
        pickle.dump(C_, f)

# Load up data
def _get_data(path):
    # Variables initialization
    W_, R_, N_, S_, M0_, M1_, M2_, Y_ = [], [], [], [], [], [], [], []
    # Loop over files
    for file in os.listdir(path):
        print(file)
        # Load file
        x_, y_ = _load_file(r'{}/{}'.format(path, file))[0]
        # Loop over samples
        for i in range(len(x_)):
            w_, r_, n_, s_, m0_, m1_, m2_ = x_[i]
            # Group data in a list per source of feature
            W_.append(w_[..., np.newaxis])
            R_.append(r_.flatten()[..., np.newaxis])
            N_.append(n_.flatten()[..., np.newaxis])
            S_.append(s_.flatten()[..., np.newaxis])
            M0_.append(m0_.flatten()[..., np.newaxis])
            M1_.append(m1_.flatten()[..., np.newaxis])
            M2_.append(m2_.flatten()[..., np.newaxis])
            Y_.append(np.array((y_))[..., np.newaxis] )
    return W_, R_, N_, S_, M0_, M1_, M2_, Y_

# Group data of a class
def _append(w_, r_, n_, s_, m_, y_, x_, label):
    if y_ == label:
        x_.append([w_, r_, n_, s_, m_, y_])
    return x_

# Form a batch
def _quanta(x_, X_, N = 410):
    if len(x_) == N:
        X_.append(x_)
        x_ = []
    return x_, X_

# Group data in batches per class
def _data_quantification(W_, R_, N_, S_, M_, Y_):
    # Variables initialization
    X_0_, X_1_, X_2_, X_3_ = [], [], [], []
    x_0_, x_1_, x_2_, x_3_ = [], [], [], []
    for w_, r_, n_, s_, m_, y_ in zip(W_, R_, N_, S_, M_, Y_):
        x_0_ = _append(w_, r_, n_, s_, m_, y_, x_0_, label = 0)
        x_1_ = _append(w_, r_, n_, s_, m_, y_, x_1_, label = 1)
        x_2_ = _append(w_, r_, n_, s_, m_, y_, x_2_, label = 2)
        x_3_ = _append(w_, r_, n_, s_, m_, y_, x_3_, label = 3)
        x_0_, X_0_ = _quanta(x_0_, X_0_)
        x_1_, X_1_ = _quanta(x_1_, X_1_)
        x_2_, X_2_ = _quanta(x_2_, X_2_)
        x_3_, X_3_ = _quanta(x_3_, X_3_)
    return X_0_, X_1_, X_2_, X_3_

# Random Selection of batches per class
def _select_quanta(X_, N = 6):
    # Define the same seed for all experiments
    np.random.seed(9888)
    return [X_[i] for i in np.random.permutation(len(X_))[:N]]

# Left one sample out for validation the rest of the data sets are for training
def _left_out_sample(i, X_0_, X_1_, X_2_, X_3_):
    # Left out the ith sample
    idx_ = np.delete(np.arange(len(X_0_)), i)
    # Select trainign set
    X_0_tr_ = [X_0_[j] for j in idx_]
    X_1_tr_ = [X_1_[j] for j in idx_]
    X_2_tr_ = [X_2_[j] for j in idx_]
    X_3_tr_ = [X_3_[j] for j in idx_]
    # the ith sample is for validation
    return X_0_tr_, X_1_tr_, X_2_tr_, X_3_tr_, X_0_[i], X_1_[i], X_2_[i], X_3_[i]

# Split qunata in training and test
def _split_quanta(X_0_, X_1_, X_2_, X_3_):
    return X_0_[:-1], X_1_[:-1], X_2_[:-1], X_3_[:-1], X_0_[-1], X_1_[-1], X_2_[-1], X_3_[-1]

# Put together a set of batches
def _merge_quanta(x_):
    # Variables initialization
    W_, R_, N_, S_, M_, Y_ = [], [], [], [], [], []
    # loop over samples
    for x_0, x_1, x_2, x_3, x_4, x_5 in x_:
        # Merge all the samples in the batch
        W_.append(x_0)
        R_.append(x_1)
        N_.append(x_2)
        S_.append(x_3)
        M_.append(x_4)
        Y_.append(x_5)
    # Concatenate all samples to from a single dataset
    W_ = np.concatenate((W_), axis = 1)
    R_ = np.concatenate((R_), axis = 1)
    N_ = np.concatenate((N_), axis = 1)
    S_ = np.concatenate((S_), axis = 1)
    M_ = np.concatenate((M_), axis = 1)
    Y_ = np.concatenate((Y_), axis = 0)
    return W_, R_, N_, S_, M_, Y_

# Get test dataset
def _get_test_data(x_0_, x_1_, x_2_, x_3_):
    # Merge all the features in each class batch
    w_0_, r_0_, n_0_, s_0_, m_0_, y_0_ = _merge_quanta(x_0_)
    w_1_, r_1_, n_1_, s_1_, m_1_, y_1_ = _merge_quanta(x_1_)
    w_2_, r_2_, n_2_, s_2_, m_2_, y_2_ = _merge_quanta(x_2_)
    w_3_, r_3_, n_3_, s_3_, m_3_, y_3_ = _merge_quanta(x_3_)
    # Concatenate data to from a single dataset
    W_ = np.concatenate((w_0_, w_1_, w_2_, w_3_), axis = 1).T
    R_ = np.concatenate((r_0_, r_1_, r_2_, r_3_), axis = 1).T
    N_ = np.concatenate((n_0_, n_1_, n_2_, n_3_), axis = 1).T
    S_ = np.concatenate((s_0_, s_1_, s_2_, s_3_), axis = 1).T
    M_ = np.concatenate((m_0_, m_1_, m_2_, m_3_), axis = 1).T
    Y_ = np.concatenate((y_0_, y_1_, y_2_, y_3_), axis = 0).T
    return W_, R_, N_, S_, M_, Y_

# Get training dataset
def _get_train_data(X_0_, X_1_, X_2_, X_3_):
    # Variables initialization
    W_, R_, N_, S_, M_, Y_ = [], [], [], [], [], []
    # Loop over each batches in each class set of the training set
    for x_0_, x_1_, x_2_, x_3_ in zip(X_0_, X_1_, X_2_, X_3_):
        # Merge all the features in each class batch
        w_0_, r_0_, n_0_, s_0_, m_0_, y_0_ = _merge_quanta(x_0_)
        w_1_, r_1_, n_1_, s_1_, m_1_, y_1_ = _merge_quanta(x_1_)
        w_2_, r_2_, n_2_, s_2_, m_2_, y_2_ = _merge_quanta(x_2_)
        w_3_, r_3_, n_3_, s_3_, m_3_, y_3_ = _merge_quanta(x_3_)
        # Append together of the batches
        W_.append(np.concatenate((w_0_, w_1_, w_2_, w_3_), axis = 1))
        R_.append(np.concatenate((r_0_, r_1_, r_2_, r_3_), axis = 1))
        N_.append(np.concatenate((n_0_, n_1_, n_2_, n_3_), axis = 1))
        S_.append(np.concatenate((s_0_, s_1_, s_2_, s_3_), axis = 1))
        M_.append(np.concatenate((m_0_, m_1_, m_2_, m_3_), axis = 1))
        Y_.append(np.concatenate((y_0_, y_1_, y_2_, y_3_), axis = 0))
    # Concatenate data to from a single dataset
    W_ = np.concatenate((W_), axis = 1).T
    R_ = np.concatenate((R_), axis = 1).T
    N_ = np.concatenate((N_), axis = 1).T
    S_ = np.concatenate((S_), axis = 1).T
    M_ = np.concatenate((M_), axis = 1).T
    Y_ = np.concatenate((Y_), axis = 0).T
    return W_, R_, N_, S_, M_, Y_

# Select Dataset features
def _get_dataset(W_, R_, N_, S_, M_, Y_):
    # Compute Statistics
    def __get_hist(X_, n_bins, range_):
        N, D = X_.shape
        Y_ = np.zeros((N, n_bins))
        # Compute Histogram for each Sample
        for i in range(N):
            Y_[i, :] = np.histogram(X_[i, :], bins = n_bins, range = range_, density = False)[0]
        return Y_
    # get weather features
    w_ = W_[:, i_w_]
    # get stats of the images range_ = [223.21, 450.58]
    r_ = __get_hist(R_, n_bins = n_bins_r, range_ = [223, x_upper])
    # Get velocity vectors features range_ = [1.4275769649957444e-5, 31.28325491941417])
    m_ = __get_hist(M_, n_bins = n_bins_m, range_ = [x_lower, 2])
    # Concatenate all selected features
    X_ = np.concatenate((w_, r_, m_), axis = 1)
    return X_, Y_

# Leave-one-out Cross-Validation Iteration
def _leave_one_out(X_0_, X_1_, X_2_, X_3_, C, degree):
    # Scores Variable Initialization
    e_ = np.zeros(len(X_0_))
    # Define Polinomial Expansion order
    _f = PolynomialFeatures(degree)
    # Loop over Samples
    for i in range(len(X_0_)):
        # split data in Training set and validation set
        X_0_tr_, X_1_tr_, X_2_tr_, X_3_tr_, X_0_val_, X_1_val_, X_2_val_, X_3_val_ = _left_out_sample(i,
                                                                                        X_0_, X_1_, X_2_, X_3_)
        # Get leave-one-out train Dataset
        W_, R_, N_, S_, M_, Y_ = _get_train_data(X_0_tr_, X_1_tr_, X_2_tr_, X_3_tr_)
        X_tr_, y_tr_ = _get_dataset(W_, R_, N_, S_, M_, Y_)
        # Get leave-one-out Validation Dataset
        W_, R_, N_, S_, M_, Y_ = _get_test_data(X_0_val_, X_1_val_, X_2_val_, X_3_val_)
        X_val_, y_val_ = _get_dataset(W_, R_, N_, S_, M_, Y_)
        # Polynomial Expasion of the training and test covariates
        X_poly_tr_  = _f.fit_transform(X_tr_)
        X_poly_val_ = _f.fit_transform(X_val_)
        # Fit and predict
        _SVM = LinearSVC(C = C, multi_class = 'ovr', dual = False, fit_intercept = False).fit(X_poly_tr_, y_tr_)
        y_val_hat_ = _SVM.predict(X_poly_val_)
        # Cross-validation scores evaluation
        e_[i] = accuracy_score(y_val_, y_val_hat_)
    # Return mean leave-one-out Score
    return np.mean(e_)

i_bins_r = int(sys.argv[1])
i_bins_m = int(sys.argv[2])
i_job    = int(sys.argv[3])
# Select weather features
# Ait Temp, Dew Point, pressure, humidity, elevation, azimuth, CSI, GSI, pyranometer, wind angle, wind magnitude
#i_w_ = [[2], [6], [8], [2, 6], [2, 8], [6, 8], [2, 6, 8]][i_job]
i_w_ = [[6], [6, 8]][i_job]
# Select images features
n_bins_r = np.linspace(3, 8, 5, dtype = int)[i_bins_r]
n_bins_m = np.linspace(3, 8, 5, dtype = int)[i_bins_m]
x_upper = 300
x_lower = 0.
# Select velocity vectors features
print(i_w_, n_bins_r, n_bins_m, x_lower, x_upper)

# Path to the data
path = r'/users/terren/atmospheric_condition_model/data/v5'
# Load up feature extraction results
W_, R_, N_, S_, M0_, M1_, M2_, Y_ = _get_data(path)
print(len(W_), len(R_), len(N_), len(S_), len(M0_), len(M1_), len(M2_), len(Y_))
# Get data in sets separate by class
X_0_, X_1_, X_2_, X_3_ = _data_quantification(W_, R_, N_, S_, M0_, Y_)
print(len(X_0_), len(X_1_), len(X_2_), len(X_3_))
# Get N batches of data for each class
X_0_tr_, X_1_tr_, X_2_tr_, X_3_tr_, X_0_ts_, X_1_ts_, X_2_ts_, X_3_ts_ = _split_quanta(X_0_ = _select_quanta(X_0_, N = 6),
                                                                                       X_1_ = _select_quanta(X_1_, N = 6),
                                                                                       X_2_ = _select_quanta(X_2_, N = 6),
                                                                                       X_3_ = _select_quanta(X_3_, N = 6))
print(len(X_0_tr_), len(X_1_tr_), len(X_2_tr_), len(X_3_tr_))
print(len(X_0_ts_), len(X_1_ts_), len(X_2_ts_), len(X_3_ts_))
# Number of Validation Paramters in each set
N_complex = 13
N_degrees = 4
# Cross-Validation Variables Initialization
complex_ = np.logspace(-3, 3, N_complex)
degrees_ = np.linspace(1, N_degrees, N_degrees, dtype = int)
# Scores Variable Initialization
scores_ = np.zeros((N_complex, N_degrees))
# Loop over model complexity
for idx_complex in range(N_complex):
    # Loop over polynomial expasion order
    for idx_degree in range(N_degrees):
        # Leave-one-out Cross-Validation
        scores_[idx_complex, idx_degree] = _leave_one_out(X_0_tr_, X_1_tr_, X_2_tr_, X_3_tr_, C = complex_[idx_complex], degree = degrees_[idx_degree])
        print(complex_[idx_complex], degrees_[idx_degree], scores_[idx_complex, idx_degree])

# Accuracy Optimal Parameters
print(np.max(scores_, axis = 0))
idx_complex, idx_degree = np.where(scores_ == np.max(scores_))
C = complex_[idx_complex][0]
degree = degrees_[idx_degree][0]
print(np.max(scores_), C, degree)

# Get leave-one-out train Dataset
W_, R_, N_, S_, M_, Y_ = _get_train_data(X_0_tr_, X_1_tr_, X_2_tr_, X_3_tr_)
X_tr_, y_tr_ = _get_dataset(W_, R_, N_, S_, M_, Y_)
print(X_tr_.shape, y_tr_.shape)
# Get leave-one-out tet Dataset
W_, R_, N_, S_, M_, Y_ = _get_test_data(X_0_ts_, X_1_ts_, X_2_ts_, X_3_ts_)
X_ts_, y_ts_ = _get_dataset(W_, R_, N_, S_, M_, Y_)
print(X_ts_.shape, y_ts_.shape)
# Define Polinomial Expansion order
_f = PolynomialFeatures(degree)
# Polynomial Expasion of the training and test covariates
X_poly_tr_ = _f.fit_transform(X_tr_)
X_poly_ts_ = _f.fit_transform(X_ts_)
# Fit and predict
_SVM = LinearSVC(C = C, multi_class = 'ovr', dual = False, fit_intercept = False).fit(X_poly_tr_, y_tr_)
y_ts_hat_ = _SVM.predict(X_poly_ts_)
# Score  evaluation
e_ = accuracy_score(y_ts_, y_ts_hat_)
print(e_)
print(confusion_matrix(y_ts_, y_ts_hat_))
