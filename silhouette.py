import numpy as np
from scipy.spatial.distance import pdist, squareform
 
def compute_silhouette(X, idx):
    '''
    Compute the silhouette coefficient for each sample with a cluster index.
    Returns array of silhouette coefficients for each sample, not the averaged value for all samples.
    Usage:
        X: N x K data matrix. N ... the number of samples. K ... the number of features.
        idx: N-length array of cluster indices for each samples. cluster indices must be started from zero.
    '''
    n_obs = X.shape[0]
    n_cl = len(set(idx))
 
    distance_matrix = squareform(pdist(X))
    cl_index = [np.flatnonzero(np.array(idx) == cl) for cl in range(n_cl)]
 
    a = np.zeros(n_obs) # average of within-cluster distances
    b = np.zeros(n_obs) # average of between-cluster distances (against the neighboring cluster)
    for i in range(n_obs):
        a[i] = np.mean( [ distance_matrix[i][target] for target in cl_index[idx[i]] if target != i ] )
        b[i] = np.min( [ np.mean(distance_matrix[i][targets]) for k,targets in enumerate(cl_index) if idx[i] != k ] )
 
    s = (b - a) / np.maximum(a,b)
    return s
 
 
def plot_silhouette(X, idx, filename):
    '''
    Plot the silhouette coefficient for each sample with a cluster index.
    Save the figure of silhouette coefficients as 'filename.png'.
    Usage:
        X: N x K data matrix. N ... the number of samples. K ... the number of features.
        idx: N-length array of cluster indices for each samples. cluster indices must be started from zero.
        filename: any string of the output file-name.
    '''
    import matplotlib.pyplot as plt
    from matplotlib import cm
    s = compute_silhouette(X, idx)
 
    order = np.lexsort((-s,idx)) # index of plotting order. Obtaining decreasing order by multiplying -1 with 's' values.
    cl_index = [ np.flatnonzero(idx[order] == k) for k in range(K) ]
    ytick = [ (np.max(ind)+np.min(ind)) / 2 for ind in cl_index ]
    ytickLabels = [ "%d"%x for x in range(K) ]
    cmap = cm.jet( np.linspace(0,1,K) ).tolist()
    cl_colors = [ cmap[i] for i in idx[order] ]
 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.barh(range(X.shape[0]), s[order], height=1.0, edgecolor='none', color=cl_colors)
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.yticks(ytick, ytickLabels)
    plt.xlabel('Silhouette Coefficient')
    plt.ylabel('Cluster')
    fig.savefig('%s.png'%filename)
 
 
if __name__ == '__main__':
    from sklearn import datasets
    from scipy.cluster.vq import kmeans2
 
    data = datasets.load_iris()
    X = data['data']
    K = 3
    C, idx = kmeans2(X, K)
    plot_silhouette(X, idx, 'silhouette_test')
