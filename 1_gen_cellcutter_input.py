import os
import pandas as pd

###############################################################################

# save directory
save_dir = '/Users/greg/projects/vae/output/1_cellcutter_input'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

###############################################################################

# SARDANA-097 single-cell data
csv_path = (
    '/Volumes/My Book/cylinter_input/clean_quant/output_3d_v2/' +
    'consensus_clustering.parquet'
    )
csv = pd.read_parquet(csv_path)

# drop cells for which there was not a consensus cluster (i.e. noisy cells)
csv = csv[csv['cluster'] != -1]

###############################################################################

# calculate a weighted random sample according to cluster size to class balance
F = 0.5
groups = csv.groupby('cluster')
sample_weights = pd.DataFrame({'weights': 1 / (groups.size() * len(groups))})
weights = pd.merge(
    csv[['cluster']], sample_weights, left_on='cluster', right_index=True
    )

csv = csv.sample(
    frac=F, replace=False, weights=weights['weights'], random_state=0, axis=0
    )
print()
print('Cells per cluster after cluster-weighted random sampling:')
print(csv.groupby('cluster').size().sort_values(ascending=False))

###############################################################################

# shuffle csv data
csv = csv.sample(frac=1.0, random_state=0)

# reserve 10% of data for testing and 10% for validation
split = round(len(csv) * 0.10)
test = csv[0:split]
validate = csv[split:split*2]
train = csv[split*2:]

# reset row indexes of each dataframe
test.reset_index(drop=True, inplace=True)
validate.reset_index(drop=True, inplace=True)
train.reset_index(drop=True, inplace=True)

###############################################################################

# save testing, validation, and training dataframes for cellcutter processing
test.to_csv(os.path.join(save_dir, 'test.csv'), index=False)
validate.to_csv(os.path.join(save_dir, 'validate.csv'), index=False)
train.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
