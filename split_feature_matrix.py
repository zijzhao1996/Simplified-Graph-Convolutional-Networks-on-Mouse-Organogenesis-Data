import os
import numpy as np
import pandas as pd
from scipy import sparse

dir_path = "/mnt/lab_data/kundaje/zijzhao/"
count = sparse.load_npz(os.path.join(dir_path, "organogenesis_mouse.npz"))
cell_annotate = pd.read_csv(os.path.join(dir_path, "cell_annotate.csv"))

cluster = cell_annotate['Main_Cluster'].unique()
cluster = cluster[~np.isnan(cluster)] #remove NaN

# split whole count matrix based on cluster
save_path = '/mnt/lab_data/kundaje/zijzhao/featmatrix_each_cluster'
for i in cluster:
    index = cell_annotate.index[cell_annotate['Main_Cluster'] == i].to_numpy()
    sparse.save_npz(os.path.join(save_path, "featmatrix_cluster%d.npz" % i), count[index,:])
    print("Already done cluster %d" %i)
