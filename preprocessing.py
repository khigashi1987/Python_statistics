import numpy as np
 
def relative_abundance(X, axis=0):
    '''
    transform a data matrix (with read-numbers etc.) into a relative abundance matrix.
    Args:
        X: the data matrix to be transformed.
        axis: if 0, transform along columns. if 1, transform along rows.
    Return:
        transformd numpy matrix.
    '''
    X = np.array(X).astype(float)
    Xr = np.rollaxis(X,axis)
    Xr /= Xr.sum(axis=0)
    return X
