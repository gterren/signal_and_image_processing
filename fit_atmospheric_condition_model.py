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

def _order_files(files_, idx_ = [2, 1, 5, 3, 6, 4, 12, 7, 21, 9, 10, 8, 13, 11, 14, 15, 16, 17, 18, 19, 20, 0]):
    index_0_ = []
    for i in idx_:
        for file_ in files_:
            if int(file_[file_.find('-') + 1:file_.find('_')]) == i:
                index_0_.append(file_)
                break
    return index_0_

# Load up data
def _get_data(path):
    # Variables initialization
    W_, R_, N_, S_, M0_, M1_, M2_, Y_ = [], [], [], [], [], [], [], []
    # Loop over files
    files_ = os.listdir(path)
    files_ = _order_files(files_)
    for file in files_:
        print(file)
        # Load file
        x_, y_ = _load_file(r'{}/{}'.format(path, file))[0]
        # Loop over samples
        for i in range(len(x_)):
            w_, r_, n_, s_, m0_, m1_, m2_ = x_[i]
            #print(r_.min(), r_.max(), n_.min(), n_.max(), s_.min(), s_.max())
            # Group data in a list per source of feature
            W_.append(w_[..., np.newaxis])
            R_.append(r_.flatten()[..., np.newaxis])
            N_.append(n_[0].flatten()[..., np.newaxis])
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
    idx_ = np.random.permutation(len(X_))
    return [X_[i] for i in idx_[:N]], [X_[i] for i in idx_[N:]]

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

# Select Dataset features
def _get_dataset(W_, R_, N_, S_, M_, Y_):
    # Compute Statistics
    def __get_stats(X_):
        # Mean, Std, Skew, and Kurtosis
        return np.array((np.mean(X_, axis = 1), np.std(X_, axis = 1), skew(X_, axis = 1), kurtosis(X_, axis = 1))).T
    # get weather features
    w_ = W_[:, i_w_]
    # get stats of the images
    r_ = __get_stats(R_)[:, i_r_]
    n_ = __get_stats(N_)[:, i_n_]
    s_ = __get_stats(S_)[:, i_s_]
    # Get velocity vectors features
    m_ = __get_stats(M_)[:, i_m_]
    # Concatenate all selected features
    X_ = np.concatenate((w_, r_, n_, s_, m_), axis = 1)
    return X_, Y_

# Save model in path with name
def _save_model(C_, path, name):
    file = r'{}/{}'.format(path, name)
    print(file)
    with open(file, 'wb') as f:
        pickle.dump(C_, f)

# Ait Temp, Dew Point, pressure, humidity, elevation, azimuth, CSI, GSI, pyranometer, wind angle, wind magnitude
i_w_ = [2]
# Select images featuresZ
# Mean, std, skew, kurtosisz
i_r_ = [0, 1, 2, 3]
i_n_ = []
i_s_ = []
# Select velocity vectors features
# Mean, std, skew, kurtosis
i_m_ = [0, 1, 2, 3]
print(i_w_, i_r_, i_n_, i_s_, i_m_)
# Path to the data
path = r'/users/terren/atmospheric_condition_model/data/v5'
# Load up feature extraction results
W_, R_, N_, S_, M0_, M1_, M2_, Y_ = _get_data(path)
print(len(W_), len(R_), len(N_), len(S_), len(M0_), len(M1_), len(M2_), len(Y_))
# Get data in sets separate by class
X_0_, X_1_, X_2_, X_3_ = _data_quantification(W_, R_, N_, S_, M0_, Y_)
# Get N batches of data for each class
X_0_tr_, X_0_ts_ = _select_quanta(X_0_, N = 5)
X_1_tr_, X_1_ts_ = _select_quanta(X_1_, N = 5)
X_2_tr_, X_2_ts_ = _select_quanta(X_2_, N = 5)
X_3_tr_, X_3_ts_ = _select_quanta(X_3_, N = 5)
print(len(X_0_tr_), len(X_1_tr_), len(X_2_tr_), len(X_3_tr_))
print(len(X_0_ts_), len(X_1_ts_), len(X_2_ts_), len(X_3_ts_))
# Accuracy Optimal Parameters
C = 316.22776601683796
degree = 1
# Get leave-one-out train Dataset
W_, R_, N_, S_, M_, Y_ = _get_train_data(X_0_tr_, X_1_tr_, X_2_tr_, X_3_tr_)
X_tr_, y_tr_ = _get_dataset(W_, R_, N_, S_, M_, Y_)
# Define Polinomial Expansion order
_f = PolynomialFeatures(degree)
# Polynomial Expasion of the training and test covariates
X_poly_tr_ = _f.fit_transform(X_tr_)
# Fit and predict
_SVM = LinearSVC(C = C, multi_class = 'ovr', dual = False, fit_intercept = False).fit(X_poly_tr_, y_tr_)
# Get  tet Dataset
W_, R_, N_, S_, M_, Y_ = _get_test_data(X_0_ts_[1], X_1_ts_[0], X_2_ts_[8], X_3_ts_[4])
X_ts_, y_ts_ = _get_dataset(W_, R_, N_, S_, M_, Y_)
# Predict
X_poly_ts_ = _f.fit_transform(X_ts_)
y_ts_hat_ = _SVM.predict(X_poly_ts_)
# Score  evaluation
print(accuracy_score(y_ts_, y_ts_hat_))
print(confusion_matrix(y_ts_, y_ts_hat_))


_save_model(C_ = [_SVM, degree], path = r'/users/terren/atmospheric_condition_model/models',
name = 'atmospheric_condition_model_v5-TM2.pkl')

# This is to analyze the performance of each batch of data
# for i in range(len(X_3_ts_)):
#     # Get leave-one-out tet Dataset
#     W_, R_, N_, S_, M_, Y_ = _merge_quanta(X_3_ts_[i])
#     X_ts_, y_ts_ = _get_dataset(W_.T, R_.T, N_.T, S_.T, M_.T, Y_.T)
#     X_poly_ts_ = _f.fit_transform(X_ts_)
#     y_ts_hat_ = _SVM.predict(X_poly_ts_)
#     # Score  evaluation
#     e_ = accuracy_score(y_ts_, y_ts_hat_)
#     print(i, e_)
#     print(confusion_matrix(y_ts_, y_ts_hat_))
