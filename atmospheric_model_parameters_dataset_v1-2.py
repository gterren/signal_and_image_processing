# Exponential Model + Lorentz function optimized indepentely

import os, pickle, datetime

import numpy as np
import matplotlib.pylab as plt

from cv2 import imread, IMREAD_UNCHANGED
from scipy import interpolate
from datetime import datetime
from scipy.optimize import fmin_l_bfgs_b

# Load Data from a pickle
def _load_data(file):
    with open(file, 'rb') as f:
        X = pickle.Unpickler(f).load()
        Y = pickle.Unpickler(f).load()
    return X, Y

# Save Data in a pickle
def _save_data(X_, file):
    with open(file, 'wb') as f:
        pickle.dump(X_, f)

def _coordinares_grid(M = 60, N = 80):
    X_, Y_ = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    Z_ = np.sqrt((X_ - 40)**2 + (Y_ - 30)**2)
    return np.concatenate([X_.flatten()[:, np.newaxis], Y_.flatten()[:, np.newaxis]], axis = 1)

# Direct radiation from the sun funcion
def _f_direct(w_, XY_, x_):
    c = 20000.
    # Unpck function parameters
    w_2, w_3, = w_
    Z_hat_ = w_2 * ( (w_3**2) / ( (XY_[:, 0] - x_[0])**2 + (XY_[:, 1] - x_[1])**2 + w_3**2 )**1.5 )[:, np.newaxis]
    Z_hat_[Z_hat_ >= c] = c
    return Z_hat_

# Atmospheric Function gradient
def _df_direct(w_, XY_, x_):
    # Unpck function parameters
    w_2, w_3, = w_
    # Gradinet Constant
    c = (XY_[:, 0] - x_[0])**2 + (XY_[:, 1] - x_[1])**2 + w_3**2
    # Calculate each parameter gradiante
    g_2_ = w_3**2 / c**(3./2.)
    g_3_ = ( w_2 * w_3 * ( 2.*(XY_[:, 0] - x_[0])**2 + 2.*(XY_[:, 1] - x_[1])**2 - w_3**2) ) / c**(5./2.)
    # And concatenate together to define the function gradient
    return np.concatenate((g_2_[:, np.newaxis], g_3_[:, np.newaxis]), axis = 1)

# Error Metric Direct Function
def _E_direct(w_, x_, z_, XY_, D):
    return np.sum(  (z_ - _f_direct(w_, XY_, x_) )**2 )

# Gradient Error Metric Direct Function
def _dE_direct(w_, x_, z_, XY_, D):
    dE_ = 2.* (_f_direct(w_, XY_, x_) - z_ )
    df_ = _df_direct(w_, XY_, x_)
    return np.sum(dE_ * df_, axis = 0)

# Scatter radiation background function
def _f_scatter(w_, XY_, x_):
    # Unpck function parameters
    w_0, w_1, = w_
    return w_0 * np.exp( (XY_[:, 1] - x_[1]) / w_1)[:, np.newaxis]

# Atmospheric Function gradient
def _df_scatter(w_, XY_, x_):
    # Unpck function parameters
    w_0, w_1, = w_
    # Gradinet Constant
    c = XY_[:, 1] - x_[1]
    # Calculate each parameter gradiante
    g_0_ = np.exp( c/w_1 )
    g_1_ = w_0 * (- c/w_1 ) * np.exp( c/w_1 )
    # And concatenate together to define the function gradient
    return np.concatenate((g_0_[:, np.newaxis], g_1_[:, np.newaxis]), axis = 1)

# Error Metric Scatter Function
def _E_scatter(w_, x_, z_, XY_, D):
    return np.sum(  (z_ - _f_scatter(w_, XY_, x_) )**2 )

# Gradient Error Metric Scatter Function
def _dE_scatter(w_, x_, z_, XY_, D):
    dE_ = 2.* (_f_scatter(w_, XY_, x_) - z_ )
    df_ = _df_scatter(w_, XY_, x_)
    return np.sum(dE_ * df_, axis = 0)

# Implementation of the line search gradient based optimization method
def _optimize(_f, _g, bounds_, args_, n_init, approx_grad):
    # Random initialization of the line search for each kernel hyper-parameter within the bounds.
    def __rand_init(bounds_):
        n_var   = len(bounds_)
        x_init_ = np.zeros(n_var)
        for i in range(n_var):
            x_init_[i] = np.random.uniform(bounds_[i][0], bounds_[i][1])
        return x_init_
    # Variables Initialization
    f_ = []
    x_ = []
    i = 0
    # Loop over no. of initializations
    while i < n_init:
        # Run the line search BFGS optimization!
        OPT_ = fmin_l_bfgs_b(_f, x0 = __rand_init(bounds_), fprime = _g, args = args_, bounds = bounds_, maxfun = 15000, approx_grad = approx_grad)
        # Save only results that converge to optima
        if not np.isnan(OPT_[1]):
            f_.append(OPT_[1])
            x_.append(OPT_[0])
            i+=1
    # Get best result of all initializations
    i = np.argmin(f_)
    return f_[i], x_[i], i

# Calculate the model error in the circumsolar area
def _inner_circumsolar_area_error(z_, z_hat_, XYZ_, x_, r = 5):
    idx_ = np.sqrt((XYZ_[:, 0] - x_[0])**2 + (XYZ_[:, 1] - x_[1])**2) < r
    return np.sqrt(np.mean((z_[idx_] - z_hat_[idx_])**2))


# Calculate the model error in the circumsolar area
def _outter_circumsolar_area_error(z_, z_hat_, XYZ_, x_, r = 5):
    idx_ = np.sqrt((XYZ_[:, 0] - x_[0])**2 + (XYZ_[:, 1] - x_[1])**2) > r
    return np.sqrt(np.mean((z_[idx_] - z_hat_[idx_])**2))

path = r'C:\Users\Guille\Desktop\troposphere_radiometry_model\data\{}'
name = 'atmospheric_dataset.pkl'
file = path.format(name)
print(file)
X_, Y_ = _load_data(file)
print(len(X_), len(Y_))
XYZ_ = _coordinares_grid(M = 60, N = 80)
print(XYZ_.shape)

# List of results Initialization
BD_ = []
BA_ = []
BW_ = []
BE_ = []
BC_ = []
BT_ = []
# loop over days
for i in range(len(X_)):
    # Variable Initialization
    DA_ = np.empty((0, 2))
    DW_ = np.empty((0, 4))
    DE_ = np.empty((0, 1))
    DC_ = np.empty((0, 1))
    DT_ = np.empty((0, 1), dtype = 'datetime64')
    day = str(X_[i][0, :][4]).split(' ')[0]
    try:
        print(day)
        # loop over samples in each day
        for ii in range(X_[i].shape[0]):
            # Extract Data and Define Dataset
            t_ = X_[i][ii, :][4]
            a_ = np.array([X_[i][ii, :][0], X_[i][ii, :][1]]).T
            x_ = np.hstack((X_[i][ii, :][2], X_[i][ii, :][3]))[:, np.newaxis]
            z_ = Y_[i][ii, :][:, np.newaxis]
            D  = z_.shape[0] * z_.shape[1]
            # Optimization Parameters and Boundaries
            OPT_ = _optimize(_E_scatter, _dE_scatter, bounds_ = [(15000., 35000.), (1., 5000)], args_ = (x_, z_, XYZ_, D), n_init = 3, approx_grad = False)
            # Get parameters of Model Approximation
            w_ = OPT_[1]
            # Model prediction for test
            z_hat_ = _f_scatter(w_, XYZ_, x_)
            g_ = z_ - z_hat_
            c  = 20000
            g_[g_ > c] = c
            # Scatter Model prediction for test
#             z_hat_ = _f_scatter(w_, XYZ_, x_)
#             g_ = z_ - z_hat_
            # Optimization Parameters and Boundaries
            OPT_ = _optimize(_E_direct, _dE_direct, bounds_ = [(100000., 1000000.), (.1, 1.)], args_ = (x_, g_, XYZ_, D), n_init = 3, approx_grad = True)
            # Get parameters of Model Approximation
            k_ = OPT_[1]
            # Direct Model prediction for test
            g_hat_ = _f_direct(k_, XYZ_, x_)
            # Calculate the model error and error in the circumsolar area
            e = _outter_circumsolar_area_error(z_, z_hat_, XYZ_, x_)
            c = _inner_circumsolar_area_error(g_, g_hat_, XYZ_, x_)
            # Diplay sample results summary
            w_ = np.concatenate((w_, k_), axis = 0)
            # Display and get best models according to error and circumsolar area error
            print('>> Date: {} O.E.: {} I.E.: {} Theta: {}'.format(t_, e, c, w_))
            # Save Model that approximate the function best
            DA_ = np.vstack((DA_, a_))
            DW_ = np.vstack((DW_, w_))
            DE_ = np.vstack((DE_, e))
            DC_ = np.vstack((DC_, c))
            DT_ = np.vstack((DT_, t_))
        print(DA_.shape, DW_.shape, DE_.shape, DC_.shape, DT_.shape)
        # Append all the data from all the days
        BD_.append(day)
        BA_.append(DA_)
        BW_.append(DW_)
        BE_.append(DE_)
        BC_.append(DC_)
        BT_.append(DT_)
        print(len(BD_), len(BA_), len(BW_), len(BE_), len(BC_), len(BT_))
    except:
        pass
    # save the Dataset
    name = r'atmospheric_model_parameters_dataset_v1-2.pkl'
    file = path.format(name)
    print(file)
    _save_data([BD_, BA_, BW_, BE_, BC_, BT_], file)
