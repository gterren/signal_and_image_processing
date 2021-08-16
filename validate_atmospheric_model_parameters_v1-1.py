# To do list:
# - Find Days in drive repository folder
# - Load files of a day:
#   * Infrared Images
#   * Pyranometer Measures
#   * Weather Data

import os, sys
import pickle

import numpy as np
import matplotlib.pylab as plt

from cv2 import imread, IMREAD_UNCHANGED
from scipy import interpolate

# To do list:
# - Define Dataset
# - Train and test samples
# - Polynimial Model Validation
# - Variables Validation
# - Error Metrics

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score


# Select as covariate: Year, Year Day, Temp air, Temp Dew, Pressure, Humidity, elevation Azimuth
def _select_dimensions(x_, idx_):
    return x_[:, idx_]

# Load Data from a pickle
def _load_data(file):
    with open(file, 'rb') as f:
        X = pickle.Unpickler(f).load()
    return X

def _split_train(X_, Y_, n_test):
    np.random.seed(0)
    idx_ = np.random.permutation(len(X_))
    X_tr_ = [X_[i] for i in idx_[:-n_test]]
    X_ts_ = [X_[i] for i in idx_[-n_test:]]
    Y_tr_ = [Y_[i] for i in idx_[:-n_test]]
    Y_ts_ = [Y_[i] for i in idx_[-n_test:]]
    return X_tr_, X_ts_, Y_tr_, Y_ts_

# Select Covariates and Regressors
def _dataset(X_, Y_, cov_index_, reg_index_):
    return X_[:, cov_index_], Y_[:, reg_index_]

def _LS(X_, y_, l = 0):
    return np.matmul(np.linalg.inv(np.matmul(X_.T, X_) + l*np.identity(X_.shape[1])*X_.shape[0]), np.matmul(X_.T, y_))

def _predict(x_, w_):
    return np.matmul(x_, w_)

# Select Covariates and Regressors for leave-one-out CV-method
def _get_leave_one_out_dataset(X_, Y_, cov_index_, reg_index_, idx_sample):
    # Get index of training days
    idx_ = np.delete(np.arange(len(X_)), idx_sample)
    # Get Training Days
    X_tr_ = [X_[i] for i in idx_]
    y_tr_ = [Y_[i] for i in idx_]
    X_tr_ = np.concatenate((X_tr_), axis = 0)
    y_tr_ = np.concatenate((y_tr_), axis = 0)
    # Get test day
    X_ts_ = X_[idx_sample]
    y_ts_ = Y_[idx_sample]
    return X_tr_[:, cov_index_], y_tr_[:, reg_index_], X_ts_[:, cov_index_], y_ts_[:, reg_index_]

# Leave-one-out Cross-Validation method
def _leave_one_out(X_tr_, Y_tr_, cov_index_, reg_index, degree, alpha):
    N_sample = len(X_tr_)
    # Define Polinomial Expansion order
    _f = PolynomialFeatures(degree)
    # Scores Variable Initialization
    e_ = np.zeros((N_sample, 2))
    # Loop over Samples
    for idx_sample in range(N_sample):

        # Get leave-one-out Dataset
        x_val_tr_, y_val_tr_, x_val_ts_, y_val_ts_ = _get_leave_one_out_dataset(X_tr_, Y_tr_,
                                                            cov_index_, reg_index_, idx_sample)
        # Polynomial Expasion of the training and test covariates
        x_val_tr_ = _f.fit_transform(x_val_tr_)
        x_val_ts_ = _f.fit_transform(x_val_ts_)

        # Fit and predict
        y_val_ts_hat_ = _predict(x_val_ts_, w_ = _LS(x_val_tr_, y_val_tr_, alpha))

        # Cross-validation scores evaluation
        e_[idx_sample, 0] = np.sqrt(mean_squared_error(y_val_ts_, y_val_ts_hat_))
        e_[idx_sample, 1] = r2_score(y_val_ts_, y_val_ts_hat_)

    return np.mean(e_, axis = 0)

# Radiometry Functionality of the Infrared Camera with Shutter
def _infrared_radiometry(I_, t_air, t_dew, p_air):
    def __cloud_height(T_cloud_, t_air, alpha):
        return (t_air - T_cloud_) / alpha
    # Transform Infrared Image Pixels Intensity to Kelvin
    def __intensity_to_kelvin(I_):
        return I_/100.0
    # Moist Adiabatic Lapse Rate
    def __MALR(T, T_d, p):
        # Earth's gravitational acceleration = 9.8076 m/s2
        g = 9.8076
        # heat of vaporization of water = 2501000 J/kg
        H_v = 2501000
        # specific gas constant of dry air = 287 J/kg·K
        R_sd = 287
        # specific gas constant of water vapour = 461.5 J/kg·K
        R_sw = 461.5
        # the dimensionless ratio of the specific gas constant of dry air to the specific gas constant for water vapour = 0.622
        epsilon = R_sd/R_sw
        # the water vapour pressure of the saturated air
        #e = epsilon * np.exp( (7.5 * T)/(273.3 + T) ) * 100 # Actual
        e = epsilon * np.exp( (7.5 * T_d)/(273.3 + T_d) ) * 100 # Saturates in hPa but tansform to Pa
        # the mixing ratio of the mass of water vapour to the mass of dry air
        r = (epsilon*e)/(p - e)
        # the specific heat of dry air at constant pressure, = 1003.5 J/kg·K
        c_pd = 1003.5
        return g * ( (R_sd * T**2) + (H_v * r * T) ) / ( (c_pd * R_sd * T**2) + (H_v**2 * r * epsilon) )
    # Celcius to Kelvin and mmHg to Pa
    def __convert_units(T, T_d, p):
        return T + 273.15, T_d + 273.15, p * 133.322
    # Unit Intensity to 16 bits Intensity that is equivalent to Kelvin degrees x 100
    k_ = __intensity_to_kelvin(I_)
    # Air and Dew Temperature from Celcius to Kelvin
    t_air, t_dew, p_air = __convert_units(t_air, t_dew, p_air)
    # Integrate Temperature Measurements to find average height of an object on the IR images
    gamma = __MALR(t_air, t_dew, p_air)
    return __cloud_height(k_, t_air, alpha = gamma)

path = r'/users/terren/atmospheric_radiometry_model/data/{}'
name = 'atmospheric_model_parameters_fitting_dataset_v1-1.pkl'
file = path.format(name)
print(file)
Cov_, Reg_, Mag_ = _load_data(file)
print(len(Cov_), len(Reg_), len(Mag_))
# Variables initialization
X_, Y_ = [], []
# Index of days with artifacts
idx_ = set(np.arange(len(Cov_), dtype = int)) ^ set([6, 9, 12, 24, 28, 31, 38, 42, 48])
Cov_ = [Cov_[i] for i in idx_]
Reg_ = [Reg_[i] for i in idx_]
Mag_ = [Mag_[i] for i in idx_]
# Loop over Covariate, regressors, and error magnitude per sample day
for x_, y_, e_ in zip(Cov_, Reg_, Mag_):
    # Select Dimentsions
    x_ = _select_dimensions(x_, idx_ = [4, 3, 6, 7, 8, 9, 10, 11])

    #x_prime_ = x_[:, [2, 3, 4]]
    #y_[:, 0] = _infrared_radiometry(y_[:, 0], x_prime_[:, 0], x_prime_[:, 1], x_prime_[:, 2])

    y_prime_ = np.diff(y_[:, 1])
    sigma = np.std(y_prime_)/5.
    idx_ = (y_prime_ < sigma ) & (y_prime_ > - sigma)

    # Append samples for training model
    X_.append(x_[1:, :][idx_, :])
    Y_.append(y_[1:, :][idx_, :])

print(len(X_), len(Y_))

X_tr_, X_ts_, Y_tr_, Y_ts_ = _split_train(X_, Y_, n_test = 10)
print(len(X_tr_), len(X_ts_), len(Y_tr_), len(Y_ts_))

idx = int(sys.argv[1])
# year day, Second of the day, air temperature, dew temperature, atmospheric presure, humidity
cov_index_ = [[0], [0, 6], [0, 6, 7]][idx]
#cov_index_ = [[2], [2, 3], [2, 3, 6], [2, 3, 6, 7]][idx]
reg_index_ = [1]
N_alphas  = 41
N_degrees = 7
print(cov_index_, reg_index_)

# Variables Initialization
alphas_  = np.logspace(-20, 20, N_alphas)
degrees_ = np.linspace(1, N_degrees, N_degrees, dtype = int)
scores_  = np.zeros((N_alphas, N_degrees, 2))

# Loop over regularization parameters
for idx_alpha_ in range(N_alphas):
    # Loop over polynomial expasion order
    for idx_degree_ in range(N_degrees):
        # Cross-Validate Parameters and get scores
        scores_[idx_alpha_, idx_degree_, :] = _leave_one_out(X_tr_, Y_tr_, cov_index_, reg_index_,
                                                             degree = degrees_[idx_degree_],
                                                             alpha = alphas_[idx_alpha_])
        print(alphas_[idx_alpha_], degrees_[idx_degree_], scores_[idx_alpha_, idx_degree_, :])

for idx_degree_ in range(N_degrees):
    # R2 Optimal Parameters
    R2_max = np.max(scores_[:, idx_degree_, 1])
    idx_alpha, idx_degree = np.where(scores_[..., 1] == R2_max)
    alpha  = alphas_[idx_alpha][0]
    degree = degrees_[idx_degree][0]
    print('R2 Val.: ', R2_max, alpha, degree)
    # MSE Optimal Parameters
    MSE_min = np.min(scores_[:, idx_degree_, 0])
    idx_alpha, idx_degree = np.where(scores_[..., 0] == MSE_min)
    alpha  = alphas_[idx_alpha][0]
    degree = degrees_[idx_degree][0]
    print('RMSE Val.: ', MSE_min, alpha, degree)

    # Get Dataset
    x_tr_, y_tr_ = _dataset(np.concatenate((X_tr_), axis = 0), np.concatenate((Y_tr_), axis = 0), cov_index_, reg_index_)
    x_tr_ = PolynomialFeatures(degree).fit_transform(x_tr_)
    # Fit Model
    w_ = _LS(x_tr_, y_tr_, alpha)
    y_tr_hat_ = _predict(x_tr_, w_)
    # Training Error
    mse_tr_ = np.sqrt(mean_squared_error(y_tr_, y_tr_hat_))
    r2_tr_  = r2_score(y_tr_, y_tr_hat_)
    print('R2 Tr.: ', r2_tr_, alpha, degree)
    print('RMSE Tr. :', mse_tr_, alpha, degree)

    # Test Variables Initialization
    mse_ts_ = np.zeros((len(X_ts_)))
    r2_ts_  = np.zeros((len(X_ts_)))
    # Loop Over training samples
    for i in range(len(X_ts_)):
        # Get Dataset
        x_ts_, y_ts_ = _dataset(X_ts_[i], Y_ts_[i], cov_index_, reg_index_)
        x_ts_ = PolynomialFeatures(degree).fit_transform(x_ts_)
        # Pridict
        y_ts_hat_ = _predict(x_ts_, w_)
        mse_ts_[i] = np.sqrt(mean_squared_error(y_ts_, y_ts_hat_))
        r2_ts_[i]  = r2_score(y_ts_, y_ts_hat_)
    # Test scores
    print('R2 Ts.: ', np.mean(r2_ts_, axis = 0), alpha, degree)
    print('RMSE Ts.: ', np.mean(mse_ts_, axis = 0), alpha, degree)
