#! /usr/bin/env python

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import networkx as nx
import pandas as pd
import matrix_code

from scipy import sparse, linalg
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random import sample
from itertools import combinations

dir_path = "/mnt/lab_data/kundaje/zijzhao/"

def load_data(num):
    """
    Load gene annotation, cell annotation as well as adjacency matrix, 
    feature matrix for the i^th cluster.

    Arguments:
    num: the i^th cluster to process

    Return: 
    gene_annotate: gene annotation info
    cell_annotate: cell annotation info 
    feature_matrix: i^th cluster gene count matrix
    adjacency matrix: i^th cluster adjacency matrix
    """

    gene_annotate = pd.read_csv(os.path.join(dir_path, "gene_annotate.csv"))
    # initialize gene weights column
    gene_annotate['logit_coeff'] = 0
    cell_annotate = pd.read_csv(os.path.join(dir_path, "cell_annotate.csv"))
    feature_matrix = sparse.load_npz(os.path.join(dir_path, 
                    "featmatrix_each_cluster/featmatrix_cluster%d.npz" % num))
    adjacency_matrix = sparse.load_npz(os.path.join(dir_path, 
                    "adjmatrix_each_cluster/adjmatrix_cluster%d.npz" % num))

    return gene_annotate, cell_annotate, feature_matrix, adjacency_matrix

def preprocess(gene_annotate, cell_annotate, feature_matrix, 
                adjacency_matrix, num, stage1, stage2):
    """
    Preprocess input files and obtain matrices necessary for SGC algorithm. (binary classification)

    Arguments:
    stage1: the former development stage of interest 
    stage2: the later development stage of interest 

    Return:
    gene_annotate: processed gene annotation info
    cell_annotate: processed cell annotation info 
    A: sparse adjacency maxtrix
    X_norm: filted and normalized sparse feature matrix
    Y: binary label numpy array (stage1: 0, stage2: 1)
    """

    # construct Y
    cell_annotate = cell_annotate.loc[cell_annotate['Main_Cluster'] == num].reset_index(drop=True)
    ind_stage1 = cell_annotate.index[cell_annotate['development_stage'] == stage1]
    ind_stage2 = cell_annotate.index[cell_annotate['development_stage'] == stage2]
    Y = np.array([0 if i in ind_stage1 else (1 if i in ind_stage2 else np.nan) 
                for i in range(cell_annotate.shape[0])])
    cell_ind = np.where(np.invert(np.isnan(Y)))[0]
    Y = Y[cell_ind]

    # construct X
    ## Preprocess I: filter those other stage cells
    X = feature_matrix
    X = X[cell_ind,:]
    cell_annotate = cell_annotate.iloc[cell_ind].reset_index(drop=True)
    ## Preoprocess II: only keep protein coding genes
    procoding_ind = gene_annotate.index[gene_annotate['gene_type']=='protein_coding'].to_numpy()
    gene_annotate = gene_annotate.loc[gene_annotate['gene_type']=='protein_coding'].reset_index(drop=True)
    X = X.transpose()[procoding_ind, :].transpose()
    ## Preprocess III: filter out no expression genes
    zeroread_ind = np.where(X.sum(axis=0)!=0)[1]
    gene_annotate = gene_annotate.iloc[zeroread_ind].reset_index(drop=True)
    X = X.transpose()[zeroread_ind, :].transpose() 
    ## Preprocess IV: filter out mito genes (genes starting with 'mt-')
    gene_name = np.array([1 if 'mt-' in i else 0 for i in np.array(gene_annotate['gene_short_name'])])
    nonmt_ind = np.where(gene_name == 0)[0]
    gene_annotate = gene_annotate.iloc[nonmt_ind].reset_index(drop=True)
    X = X.transpose()[nonmt_ind, :].transpose() 
    ## Preprocess V: Normalization & filter out low expression variance genes
    ## only keep genes with variance above 10% quantile
    X_norm = matrix_code.deviance_residuals(X)
    score = perReadScore(X, X_norm)
    quantile = np.quantile(score, 0.1)
    abovequant_ind = np.where(score >= quantile)[1]
    gene_annotate = gene_annotate.iloc[abovequant_ind].reset_index(drop=True)
    X_norm = X_norm.transpose()[abovequant_ind, :].transpose()

    # construct A
    A = adjacency_matrix
    A = A[cell_ind,:]
    A = A.transpose()[cell_ind,:].transpose()

    return gene_annotate, cell_annotate, A, X_norm, Y


def perReadScore(X, X_norm, axis=0):
    """
    Helper function for preprocess.
    sums of squares of normalized column entries / num of reads
    """

    num_reads = X.sum(axis=0)
    return X_norm.power(2).sum(axis=axis)/num_reads 


def plot_graph(A):
    """
    Plot the graph based on the adjacency matrix.
    """

    G = nx.from_scipy_sparse_matrix(A)
    nx.draw(G)
    plt.savefig('./graph.png', dpi=300, bbox_inches = 'tight')
    plt.show()


def heatmap(L, labels):
    """
    Generate a heatmap based on a list.
    """

    acc = np.ones((5, 5));
    acc[0,1]=acc[1,0]=L[0]; acc[0,2]=acc[2,0]=L[1];
    acc[0,3]=acc[3,0]=L[2]; acc[0,4]=acc[4,0]=L[3];
    acc[1,2]=acc[2,1]=L[4]; acc[1,3]=acc[3,1]=L[5];
    acc[1,4]=acc[4,1]=L[6]; acc[2,3]=acc[3,2]=L[7];
    acc[2,4]=acc[4,2]=L[8]; acc[3,4]=acc[4,3]=L[9];

    acc = pd.DataFrame(data=acc, columns=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(acc, annot=True, cmap="YlGnBu",
        xticklabels=acc.columns,
        yticklabels=acc.columns)
    plt.savefig(os.path.join(dir_path, './heatmap.png'), 
                dpi=300, bbox_inches = 'tight')
    plt.show()


def SimGraphConv(A, X, Y, k, penalty='l2'):
    """
    Simple Graph Convolution Algorithm.
    
    Arguments:
    A: Sparse adjacency matrix [n, n] (n is the number of nodes)
    X: Sparse feature matrix [n, d] (d is the number of features)
    Y: Numpy array with labels [n,1]
    k: number of layers
    penalty: 'l1', 'l2' specify the norm used in the penalization.

    Return:
    Y: Prediction [n, c] Y[i,j] denotes the the prob of node i belongs to class j
    """

    I = sparse.eye(A.shape[0])
    A_hat = A + I
    D_hat = np.asarray(A_hat.sum(axis=0)).astype(np.float64)[0]
    assert((D_hat>0).all())
    invsqrt = lambda x: x**(-0.5)
    D_hat_invsqrt = sparse.diags(invsqrt(D_hat))
    S = D_hat_invsqrt.dot(A_hat).dot(D_hat_invsqrt)
    train_ind, test_ind = train_test_split(np.arange(X.shape[0]), train_size=0.7, test_size=0.3)
    X_train = (S**k)[train_ind,:].dot(X)
    Y_train = Y[train_ind]
    X_test = (S**k)[test_ind,:].dot(X)
    Y_test = Y[test_ind]

    logfit = LogisticRegressionCV(cv=2, penalty='l2', solver='liblinear', random_state=1,
                                    max_iter=100).fit(X_train, Y_train)
    return logfit.score(X_train,Y_train),logfit.score(X_test,Y_test), logfit.coef_


if __name__ == '__main__':
    develop_stage = [9.5, 10.5, 11.5, 12.5, 13.5]
    num = 30
    k = 2
    l1_result = []
    # only need load once at the beginning
    gene_annotate, cell_annotate, feature_matrix, adjacency_matrix = load_data(num) 
    print("Load data sucessfully.")
    print("--------------------------------------------")
    for subset in combinations(develop_stage, 2):
        print("Stage1: %.01f, stage2: %.01f:" % (subset[0], subset[1]))
        gene_annot, cell_annot, A, X_norm, Y = preprocess(gene_annotate, cell_annotate, feature_matrix, 
                                                        adjacency_matrix, num, subset[0], subset[1])

        start = time.process_time()
        result = SimGraphConv(A,X_norm,Y,k)
        print("Training accuracy : %.04f, test accuracy: %.04f." % 
            (result[0], result[1]))
        l1_result.append(result[1])
        gene_annot['logit_coeff'] = gene_annot['logit_coeff'].values + result[2].squeeze()
        gene_annot = gene_annot.sort_values(by=['logit_coeff'], ascending=False)
        gene_annot = gene_annot.loc[gene_annot['logit_coeff'] > 0] 
        gene_annot.to_csv(os.path.join(dir_path, 
                            'geneweight_result/geneweight_%s_%s.csv'%(subset[0], subset[1])), index=False)
        end = time.process_time()
        print("Running time: %.02f seconds." %(end-start))

        print("--------------------------------------------")
    heatmap(l1_result, develop_stage)
    print("All work done.")

