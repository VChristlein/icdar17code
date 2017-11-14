import numpy as np
import sys
import string
np_cv_types = {'u': np.dtype(np.uint8),
               'c': np.dtype(np.int8),
               'w': np.dtype(np.uint16),
               's': np.dtype(np.int16),
               'i': np.dtype(np.int32),
               'l': np.dtype(np.int64),
               'f': np.dtype(np.float32),
               'd': np.dtype(np.float64)}

def readOcvmb(filename):
    """ 
    read a opencv matrix binary file
    Returns: 
        numpy matrix
    """ 
    try: 
        f = open(filename, 'rb')
    except IOError:
        filename = filter(lambda x: x in string.printable, filename)
        f = open(filename, 'rb') 

    ocv_type = chr(np.fromfile(f, np.uint8, 1)[0])
    np_type = np_cv_types[ocv_type]
    ocv_rows = np.fromfile(f, np.int32, 1)[0]         
    ocv_cols = np.fromfile(f, np.int32, 1)[0] 
    ocv_chans = np.fromfile(f, np.int32, 1)[0]
    n_vals = ocv_rows * ocv_cols * ocv_chans
    np_mat = np.fromfile(f, np_type, n_vals)
    if ocv_chans > 1:
        np_mat = np_mat.reshape((ocv_rows,ocv_cols,ocv_chans))
    else:
        np_mat = np_mat.reshape((ocv_rows,ocv_cols))

    f.close()

    return np_mat

if __name__ == '__main__':
    np_mat = readOcvmb(sys.argv[1])
    print(np_mat[50])
    print(np_mat.dtype)
    print(np_mat.shape)
    print(np_mat.ndim)  
    ind = np.argmax(np_mat, axis=1)
    v = np.bincount(ind, minlength=100) #.reshape(1,-1)
    print(v)
    print(float(v[0]) / np.sum(v))
    print('{} {}'.format(np_mat[np_mat==255].sum(),np.sum(np_mat)))
    if np_mat[np_mat==255].sum() == np.sum(np_mat) or\
       np_mat[np_mat==1].sum() == np_mat.sum():
        print('matrix is binary')
    else:
        print('matrix is not binary')

