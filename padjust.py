import numpy as np
 
def adjust(p, method='bonferroni'):
    '''
    Usage:
        pvalues.adjust(p, method='bonferroni')
        returns p-values adjusted using one of several methods.
        p ... numeric vector of p-values (possibly with NAs)
        method ... correction method
            bonferroni: Use Bonferroni method to control the family-wise error rate strictly.
            BH: Use method of Benjamini and Hochberg to control the false discovery rate.
            fdr: same as 'BH'.
    '''
    try:
        p = np.array(p).astype(float)
        n = len(p)
    except ValueError:
        print 'Error: input p-values contain invalid string elements.'
        quit()
    # remove "n.a." values from input vector
    p0 = p
    not_na = ~np.isnan(p)
    p = p[not_na]
    lp = len(p)
    if lp <= 1:
        return p0
    if method == 'bonferroni':
        p0[not_na] = np.fmin(1., n*p)
    elif method == 'BH' or method == 'fdr':
        i = np.arange(lp+1)[:0:-1]
        o = np.argsort(p)[::-1]
        ro = np.argsort(o)
        p0[not_na] = np.fmin(1., np.minimum.accumulate(float(n) / i.astype(float) * p[o]))[ro]
 
    return p0
