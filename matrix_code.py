import numpy as np, scipy as sp, pandas as pd, time, os, glob, anndata, sklearn
import scanpy as sc, anndata


# ===========================================================
# =============== Input feature normalization ===============
# ===========================================================

# Input: a sparse matrix
def sparse_variance(data, axis=0):
    sqmat = data.power(2).mean(axis=axis)
    return np.ravel(sqmat - np.square(np.ravel(data.mean(axis=axis))))


# Returns: a sparse matrix.
def deviance_residuals(rawdata, sqrt=True, binomial=False):
    """
    Calculate pointwise deviance residual normalization of count data.
    """
    rawdata = sp.sparse.csr_matrix(rawdata)
    signal_per_cell = np.ravel(rawdata.sum(axis=1))
    null_gene_abund = np.ravel(rawdata.sum(axis=0))/rawdata.sum()
    logmat = rawdata.T.multiply(1.0/signal_per_cell).T.multiply(1.0/null_gene_abund)
    logmat.data = np.log(logmat.data)
    dev_contribs = 2*logmat.multiply(rawdata)
    if binomial:
        rho = rawdata.T.multiply(1.0/signal_per_cell).T.toarray()
        comp_xent = np.multiply(1 - rho, np.log(1 - rho) - np.log(1 - null_gene_abund))
        comp_dev_contribs = np.multiply(comp_xent.T, 2*signal_per_cell).T
        contribs = dev_contribs + comp_dev_contribs
        if sqrt:
            sgn_logmat = np.sign(contribs)
            return np.multiply(np.sqrt(contribs), sgn_logmat)
        else:
            return contribs
    else:    # multinomial
        if sqrt:    # Low-memory for <= 1e8 entries
            sgn_logmat = logmat._with_data(np.sign(logmat.data), copy=True)
            return dev_contribs.multiply(sgn_logmat).sqrt().multiply(sgn_logmat)
        else:
            return dev_contribs



# ======================================================
# =============== kNN graph construction ===============
# ======================================================

def calc_nbrs_exact(raw_data, k=1000):
    """
    Calculate list of `k` exact Euclidean nearest neighbors for each point.
    
    Parameters
    ----------
    raw_data: array of shape (n_samples, n_features)
        Input dataset.
    Returns
    -------
    nbr_list_sorted: array of shape (n_samples, n_neighbors)
        Indices of the `n_neighbors` nearest neighbors in the dataset, for each data point.
    """
    a = sklearn.metrics.pairwise_distances(raw_data)
    nbr_list_sorted = np.argsort(a, axis=1)[:, 1:]
    return nbr_list_sorted[:, :k]



# =================================================================
# =============== Clustering / unsupervised methods ===============
# =================================================================

def compute_coclustering(
    fit_data, 
    num_clusters=1, 
    tol_bicluster=0.005,  # sparsity otherwise annoyingly causes underflows w/ sklearn
    minibatch=True, 
    sklearn_mode=True, 
    random_seed=0
):
    """
    This wrapper for sklearn coclustering should handle dense and sparse matrices.
    """
    if num_clusters == 1:
        num_clusters = min(fit_data.shape[0], 5)
    if not issparse(fit_data):
        sklearn_mode = True
        fit_data = fit_data + tol_bicluster
    if sklearn_mode:
        model = SpectralCoclustering(n_clusters=num_clusters, random_state=random_seed, mini_batch=minibatch)
        model.fit(fit_data)
        ordered_rows = np.argsort(model.row_labels_)
        ordered_cols = np.argsort(model.column_labels_)
        return (ordered_rows, ordered_cols, model.row_labels_[ordered_rows], model.column_labels_[ordered_cols])
    else:
        pass


def compute_clustering(
    fit_data, 
    num_clusters=1, 
    tol_bicluster=0.005,  # sparsity otherwise annoyingly causes underflows w/ sklearn
    minibatch=True, 
    sklearn_mode=True, 
    random_seed=0
):
    """
    This wrapper for sklearn spectral clustering should handle dense and sparse matrices.
    """
    pass