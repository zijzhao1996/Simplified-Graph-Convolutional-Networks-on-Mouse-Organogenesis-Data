# summarize pair-wise gene weights 
# and find those top significant genes which could be causal

import os
import numpy as np
import pandas as pd
from itertools import combinations
from functools import reduce

def combine_str(x):
    """
    Reduce and extract gene_name and gene_type.
    """
    if 0 in name:
    	name = list(set(x.values))
    name.remove(0)
    return name[0]

if __name__ == '__main__':
	dir_path = "/mnt/lab_data/kundaje/zijzhao/"

	develop_stage = [9.5, 10.5, 11.5, 12.5, 13.5]
	dic = {}
	for subset in combinations(develop_stage, 2):
	    dic["weight_{0}_{1}".format(subset[0], subset[1])]= pd.read_csv(os.path.join(dir_path, 
	                                            'geneweight_result/geneweight_%s_%s.csv'%(subset[0], subset[1])))
	data_frames = list(dic.values())
	df_merged = reduce(lambda left,right: pd.merge(left, right, on='gene_id', how='outer').fillna(0), data_frames)

	# combine all 'logit_coeff_x', 'logit_coeff_y' columns and add them
	merged_logit_coeff_x = df_merged['logit_coeff_x'].groupby(
							df_merged['logit_coeff_x'].columns, axis=1).sum().to_numpy()
	merged_logit_coeff_y = df_merged['logit_coeff_y'].groupby(
							df_merged['logit_coeff_y'].columns, axis=1).sum().to_numpy()
	merged_logit_coeff = merged_logit_coeff_x + merged_logit_coeff_y
	# combine all 'gene_short_name_x' and 'gene_short_name_y' columns
	gene_short_name = df_merged[['gene_short_name_x','gene_short_name_y']].apply(combine_str, axis=1)
	# combine all 'gene_type_x' and 'gene_type_y' columns
	gene_type = df_merged[['gene_type_x','gene_type_y']].apply(combine_str, axis=1)

	# reconstruct the merged df with the orginal first three columns and merged_logit_coeff
	df = pd.DataFrame({'gene_id':df_merged.iloc[:,0].to_numpy(),
	                   'gene_type': gene_type.to_numpy(),
	                   'gene_short_name': gene_short_name.to_numpy(),
	                   'merged_logit_coeff':merged_logit_coeff.squeeze()})
	df = df.sort_values(by=['merged_logit_coeff'], ascending=False).reset_index(drop=True)
	pd.DataFrame.to_csv(df, os.path.join(dir_path, 'merged_weights.csv'), sep=',', index=False)

	# valid causal genes in the original paper
	valid_list = ['Shh', 'Ntn1', 'Slit1', 'Spon1', 'Tox2', 'Stxbp6', 'Schip1', 'Frmd4b']
	for item in valid_list:
		if item in df['gene_short_name'].values:
			print("find %s gene:" % item)
			print(df.loc[df['gene_short_name']==item])
		else:
			print("Not find %s gene." % item)

