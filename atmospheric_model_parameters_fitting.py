# To do list:
# - Define Dataset
# - Train and test samples
# - Polynimial Model Validation
# - Variables Validation
# - Error Metrics
# - Find Days in drive repository folder
# - Load files of a day:
#   * Infrared Images
#   * Pyranometer Measures
#   * Weather Data

import os, pickle, sys

import numpy as np
import matplotlib.pylab as plt

from cv2 import imread, IMREAD_UNCHANGED
from scipy import interpolate
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

# Load Data from a pickle
def _load_data(file):
    with open(file, 'rb') as f:
        X = pickle.Unpickler(f).load()
    return X

# Split dataset in Training and Test
def _split_train(X_, Y_, n_test):
    X_tr_ = X_[:-n_test]
    X_ts_ = X_[-n_test:]
    Y_tr_ = Y_[:-n_test]
    Y_ts_ = Y_[-n_test:]
    return X_tr_, X_ts_, Y_tr_, Y_ts_

# Leave-One-out Slit of training dataset for cross-validations
def _split_validation(X_, Y_, idx_):
    # duplicate varibles to not overwrite varible outside the function
    X_tr_ = X_.copy()
    Y_tr_ = Y_.copy()
    # Select validation Sample
    X_ts_ = X_tr_[idx_]
    Y_ts_ = X_tr_[idx_]
    # Remove Validation Samples
    X_tr_.pop(idx_)
    Y_tr_.pop(idx_)
    # Training Dataset in matrix from
    X_tr_ = np.concatenate(X_tr_, axis = 0)
    Y_tr_ = np.concatenate(Y_tr_, axis = 0)
    return X_tr_, X_ts_, Y_tr_, Y_ts_

# Select Covariates and Regressors
def _dataset(X_, Y_, cov_index_, reg_index_):
    return X_[:, cov_index_], Y_[:, reg_index_]

# Ridge Regression Least Squares Fitting
def _LS(X_, y_, l = 0):
    return np.matmul(np.linalg.inv(np.matmul(X_.T, X_) + l*np.identity(X_.shape[1])*X_.shape[0]), np.matmul(X_.T, y_))

# Ridge Regression Predict
def _predict(x_, w_):
    return np.matmul(x_, w_)

# Leave-out-out Cross-Validation loop function
def _CV(X_tr_, Y_tr_, degree, alpha):
    # Variables Initialization
    e_ = np.zeros((len(X_tr_), 2))
    # loop over left-out-sample
    for idx_val_ in range(len(X_tr_)):
        # Split dataset in training and validation
        X_val_tr_, X_val_ts_, Y_val_tr_, Y_val_ts_ = _split_validation(X_tr_, Y_tr_, idx_val_)
        # Dataset in Matrix from
        X_val_tr_, Y_val_tr_ = _dataset(X_val_tr_, Y_val_tr_, cov_index_, reg_index_)
        X_val_ts_, Y_val_ts_ = _dataset(X_val_ts_, Y_val_ts_, cov_index_, reg_index_)
        # Define Polynimial Transformation
        _f = PolynomialFeatures(degree)
        # Dataset Polynomial Transformation
        X_val_tr_ = _f.fit_transform(X_val_tr_)
        X_val_ts_ = _f.fit_transform(X_val_ts_)
        # Ridge Regression LS-fit
        w_ = _LS(X_val_tr_, Y_val_tr_, alpha)
        # Ridge Regression Validation prediction
        Y_val_ts_hat_ = _predict(X_val_ts_, w_)
        # Error Metrics Evaluation
        e_[idx_val_, 0] = mean_squared_error(Y_val_ts_, Y_val_ts_hat_, multioutput = 'uniform_average')
        e_[idx_val_, 1] = r2_score(Y_val_ts_, Y_val_ts_hat_, multioutput = 'uniform_average')
    # Return Average leave-oe-out Error
    return np.mean(e_, axis = 0)

# Approximate the Parameters of the models with less error
def _select_samples(X_, Y_, e_, N_samples, tau = None):
    m_   = np.sqrt(e_[:, 0]**2 + e_[:, 1]**2)
    idx_ = np.argsort(m_)[:N_samples]
    return X_[idx_, :], Y_[idx_, :]

# Select as covariate: Year, Year Day, Temp air, Temp Dew, Pressure, Humidity, elevation Azimuth
def _select_dimensions(x_):
    return x_[:, [4, 3, 6, 7, 8, 9, 10, 11]]

# Select Samples and Dimensions
def _select_sample_dimension(Cov_, Reg_, Mag_, N_samples):
    # Variables initialization
    X_ = []
    Y_ = []
    # Loop over Covariate, regressors, and error magnitude per sample day
    for x_, y_, e_ in zip(Cov_, Reg_, Mag_):
        # Select Samples with smallest error
        x_, y_ = _select_samples(x_, y_, e_, N_samples)
        # Select Dimentsions
        x_ = _select_dimensions(x_)
        #print(x_.shape, y_.shape)
        # Append samples for training model
        X_.append(x_)
        Y_.append(y_)
    #print(len(X_), len(Y_))
    return X_, Y_

idx = int(sys.argv[1])
# Define Covariates and Regressiors for Experiment
cov_index_list_ = [[0, 1, 2, 3, 4, 5, 7]]
cov_index_ = cov_index_list_[idx]
reg_index_ = [0, 1, 2, 3]
path = r'C:\Users\Guille\Desktop\infrared_radiometry\atmospheric_model\{}'
name = 'atmospheric_model_parameters_fitting_dataset.pkl'
file = path.format(name)
Cov_, Reg_, Mag_ = _load_data(file)
print(len(Cov_), len(Reg_), len(Mag_))
# Select Samples and Dimensions for the dataset
X_, Y_ = _select_sample_dimension(Cov_, Reg_, Mag_, N_samples = 50)
# Split dataset in Training and Test
X_tr_, X_ts_, Y_tr_, Y_ts_ = _split_train(X_, Y_, n_test = 5)
print(len(X_tr_), len(X_ts_), len(Y_tr_), len(Y_ts_))
# Number of Regularization parameters and polynomial degree to cross-validate
N_alphas  = 5
N_degrees = 5
# Constant Initialization
alphas_  = np.logspace(-15, 15, N_alphas)
degrees_ = np.linspace(1, N_degrees, N_degrees, dtype = int)
# Variables Initialization
scores_  = np.zeros((N_alphas, N_degrees, 2))
# loop over regularization parameters
for idx_alpha_ in range(N_alphas):
    # Loop over polynomial degress
    for idx_degree_ in range(N_degrees):
        # Leave-one-out Cross-Validation of this combination of parameters
        scores_[idx_alpha_, idx_degree_, :] = _CV(X_tr_, Y_tr_, degrees_[idx_degree_], alpha = alphas_[idx_alpha_])
        # Display regularization parameters, polynomimal degree, and Score obtaine in this Evaluation
        #print(alphas_[idx_alpha_], degrees_[idx_degree_], scores_[idx_alpha_, idx_degree_, :])
# Get Best Parameters according to CV-MSE
idx_alpha, idx_degree = np.where(scores_[..., 0] == np.min(scores_[..., 0]))
print(np.min(scores_[..., 0]), alphas_[idx_alpha], degrees_[idx_degree])
