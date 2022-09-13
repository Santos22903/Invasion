import numpy as np
import pandas as pd

def coord_to_mat(coord_array,lattice_size):
    zero_arr = np.zeros((lattice_size,lattice_size),dtype=int)
    np.add.at(zero_arr, tuple(coord_array.T), 1)
    return zero_arr

def mat_to_coord(lattice_mat):
    lattice_mat = lattice_mat.astype(int)
    idx = np.transpose(np.nonzero(lattice_mat))
    coord_out = np.repeat(idx,lattice_mat[tuple(idx.T)],axis=0)
    coordDF = pd.DataFrame(data=coord_out, columns = ['x','y'])
    return coordDF
