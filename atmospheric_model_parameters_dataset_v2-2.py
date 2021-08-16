# Order 3 Polynomial Model + Lorentz function optimized together

import os, pickle, datetime, warnings

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
def _f_direct(w_2, w_3, XY_, x_):
    return w_2 * ( (w_3**2) / ( (XY_[:, 0] - x_[0])**2 + (XY_[:, 1] - x_[1])**2 + w_3**2 )**1.5 )[:, np.newaxis]

# Scatter radiation background function
def _f_scatter(w_0, w_1, w_2, XY_, x_):
    X_ = XY_[:, 1][:, np.newaxis]
    return w_2*X_**3 + w_1*X_**2 + w_0*X_

# Atmospheric Funtion
def _f(w_, b, XY_, x_):
    # Unpck function parameters
    w_0, w_1, w_2, w_3, w_4 = w_
    # Evaluation of the scatter radiation function
    f_1 = _f_scatter(w_0, w_1, w_2, XY_, x_)
    # Evaluation of the direct radiation function
    f_2 = _f_direct(w_3, w_4, XY_, x_)
    # Adding together both fuctions to obtained the atmospheric function
    Z_hat_ = f_1 + f_2 + b
    Z_hat_[Z_hat_ > 45057.] = 45057
    return Z_hat_

# Atmospheric Function gradient
def _g(w_, XY_, x_):
    # Unpck function parameters
    w_0, w_1, w_2, w_3 = w_
    # Gradinet Constant
    _h = (XY_[:, 0] - x_[0])**2 + (XY_[:, 1] - x_[1])**2 + w_3**2
    # Calculate each parameter gradiante
    g_0_ = np.exp( (XY_[:, 1] - x_[1])/w_1 )
    g_1_ = - w_0 * ( (XY_[:, 1] - x_[1])/w_1**2 ) * np.exp( (XY_[:, 1] - x_[1])/w_1 )
    g_2_ = w_3**2 / _h**1.5
    g_3_ = w_2 * w_3 * (2*(XY_[:, 0] - x_[0])**2 + 2*(XY_[:, 1] - x_[1])**2 - w_3**2) / _h**2.5
    # And concatenate together to define the function gradient
    return np.concatenate((g_0_[:, np.newaxis], g_1_[:, np.newaxis], g_2_[:, np.newaxis], g_3_[:, np.newaxis]), axis = 1)

# RMSE
def _E(w_, x_, z_, XY_, D):
    b = z_.min()
    return np.sqrt( np.mean(  (z_ - _f(w_, b, XY_, x_))**2 ) )

# Derivatif of the RMSE
def _dE(w_, x_, z_, XY_, D):
    # Evaluation of the Atmoshperic function
    z_hat_  = _f(w_, XY_, x_)
    # Evaluation of the Atmospheric function gradient
    dZ_hat_ = _g(w_, XY_, x_)
    # Definiton of the the error function gradient
    E_ = 2*dZ_hat_ - 2*z_*dZ_hat_
    return np.sum(E_, axis = 0)

# Implementation of the line search gradient based optimization method
def _optimize(_f, _g, bounds_, args_, n_init):
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
        OPT_ = fmin_l_bfgs_b(_f, x0 = __rand_init(bounds_), fprime = _g, args = args_,
                                     bounds = bounds_, maxfun = 15000, approx_grad = True)
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


# Paths and files name Definiton
path = r'C:\Users\Guille\Desktop\troposphere_radiometry_model\data\{}'
name = 'atmospheric_dataset.pkl'
file = path.format(name)
print(file)

# Load-up Dataset
X_, Y_ = _load_data(file)
print(len(X_), len(Y_))
XYZ_ = _coordinares_grid(M = 60, N = 80)
print(XYZ_.shape)

# List of results Initialization
BD_ = []
BT_ = []
BA_ = []
BW_ = []
BB_ = []
BE_ = []
BC_ = []
# Optimization Parameters Boundaries
bounds_ = [(-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (1e5, 1e6), (.1, 1.)]
# loop over days
for i in range(len(X_)):
    # Variable Initialization
    day = str(X_[i][0, :][4]).split(' ')[0]
    DT_ = np.empty((0, 1), dtype = 'datetime64')
    DA_ = np.empty((0, 2))
    DW_ = np.empty((0, 5))
    DB_ = np.empty((0, 1))
    DE_ = np.empty((0, 1))
    DC_ = np.empty((0, 1))
    try:
        print(day)
        # loop over samples in each day
        for ii in range(0, X_[i].shape[0], 3):
            # Extract Data and Define Dataset
            t_ = X_[i][ii, :][4]
            a_ = np.array([X_[i][ii, :][0], X_[i][ii, :][1]]).T
            x_ = np.hstack((X_[i][ii, :][2], X_[i][ii, :][3]))[:, np.newaxis]
            z_ = Y_[i][ii, :][:, np.newaxis]
            D  = z_.shape[0] * z_.shape[1]
            # Optimiza Model
            OPT_ = _optimize(_E, _dE, bounds_, args_ = (x_, z_, XYZ_, D), n_init = 3)
            # Get Results of Model Approximation
            e  = OPT_[0]
            w_ = OPT_[1]
            b  = z_.min()
            # Model prediction for test
            z_hat_ = _f(w_, b, XYZ_, x_)
            # Calculate the model error in the circumsolar area
            e = _outter_circumsolar_area_error(z_, z_hat_, XYZ_, x_)
            c = _inner_circumsolar_area_error(z_, z_hat_, XYZ_, x_)
            # Diplay sample results summary
            print('>> Date: {} Theta: {} Beta: {} O.E.: {} I.E.: {}'.format(t_, w_, b, e, c))
            # Save Model that approximate the function best
            DT_ = np.vstack((DT_, t_))
            DA_ = np.vstack((DA_, a_))
            DW_ = np.vstack((DW_, w_))
            DB_ = np.vstack((DB_, b))
            DE_ = np.vstack((DE_, e))
            DC_ = np.vstack((DC_, c))
        print(DT_.shape, DA_.shape, DW_.shape, DB_.shape, DE_.shape, DC_.shape)
        # Append all the data from all the days
        BD_.append(day)
        BT_.append(DT_)
        BA_.append(DA_)
        BW_.append(DW_)
        BB_.append(DB_)
        BE_.append(DE_)
        BC_.append(DC_)
        print(len(BD_), len(BT_), len(BA_), len(BW_), len(BB_), len(BE_), len(BC_))
    except:
        pass
    # save the Dataset
    name = r'atmospheric_model_parameters_dataset_v2-2.pkl'
    file = path.format(name)
    print(file)
    _save_data([BD_, BT_, BA_, BW_, BB_, BE_, BC_], file)
