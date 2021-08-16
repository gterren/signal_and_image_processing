import pickle
import numpy as np

from mpi4py import MPI

# Transfsorm Velocity Vectors in Cartenian to Polar Coordiantes
def _cart_to_polar(x, y):
    # Vector Modulus
    psi_ = np.nan_to_num(np.sqrt(x**2 + y**2))
    # Vector Direction
    phi_ = np.nan_to_num(np.arctan2(y, x))
    # Correct for domain -2pi to 2pi
    phi_[phi_ < 0.] += 2*np.pi
    return psi_, phi_

# Load all variable in a pickle file
def _load_file(name):
    def __load_variable(files = []):
        while True:
            try:
                files.append(pickle.load(f))
            except:
                return files
    with open(name, 'rb') as f:
        files = __load_variable()
    return files

# Group down together the entired dataset in predictions and covariates
def _save_file(X_, name):
    with open(name, 'wb') as f:
        pickle.dump(X_, f)
    print(name)

# Get MPI node information
def _get_node_info(verbose = False):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    if verbose:
        print('>> MPI: Name: {} Rank: {} Size: {}'.format(name, rank, size) )
    return int(rank), int(size), comm

# Define a euclidian frame of coordenate s
def _euclidian_coordiantes(N_x, N_y):
    return np.meshgrid(np.linspace(0, N_x - 1, N_x), np.linspace(0, N_y - 1, N_y))


__all__ = ['_load_file', '_save_file', '_get_node_info', '_euclidian_coordiantes', '_cart_to_polar']
