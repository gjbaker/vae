import os
import numpy as np
import pandas as pd
from tifffile import imread
import matplotlib.pyplot as plt
import pickle

# save directory
save_dir = (
    '/Users/greg/projects/vae_sardana-097/4_feature_preprocessing_selections'
    )
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

markers = pd.read_csv(
    '/Volumes/My Book/cylinter_input/sardana-097/markers.csv'
    )

cellcutter_markers = [
    'anti_CD3', 'anti_CD45RO', 'Keratin_570', 'aSMA_660', 'CD4_488', 'CD45_PE',
    'PD1_647', 'CD20_488', 'CD68_555', 'CD8a_660', 'CD163_488', 'FOXP3_570',
    'PDL1_647', 'Ecad_488', 'Vimentin_555', 'CDX2_647', 'LaminABC_488',
    'Desmin_555', 'CD31_647', 'PCNA_488', 'CollagenIV_647'
    ]

###############################################

numRows = 4
numColumns = 6
grid_dims = (numRows, numColumns)

# initialize figure canvas
fig_orig = plt.figure(figsize=(12, 8.5))
fig_log = plt.figure(figsize=(12, 8.5))
fig_clip = plt.figure(figsize=(12, 8.5))

# loop over cellcutter channels
cutoffs = {}
for e, marker in enumerate(cellcutter_markers):

    print(marker)

    # get channel number from markers.csv
    channel_number = markers['channel_number'][
        markers['marker_name'] == marker].values[0]

    # read channel
    img = imread(
      '/Volumes/My Book/cylinter_input/sardana-097/tif/WD-76845-097.ome.tif',
      key=channel_number-1)

    # log-transform image
    log_img = np.log10(img, where=(img != 0))

    # specify lower and upper percentile cutoffs
    lower_cutoff_log = np.percentile(log_img.ravel(), 0.17)
    upper_cutoff_log = np.percentile(log_img.ravel(), 99.99)

    # add channel cutoffs to dict
    cutoffs[marker] = (lower_cutoff_log, upper_cutoff_log)

    # scale 0.17th and 99.99th percentile between 0 and 1
    # Note: this will cause outlier pixels below the 0.1th percentile and above
    # the 99.9th to take values <0 and >1, respectively
    rescaled_log_img = (
        (((1-0)*(log_img.ravel()-lower_cutoff_log)) /
         (upper_cutoff_log-lower_cutoff_log)
         ) + 0).reshape(log_img.shape)

    # clip outliers to lower and upper percentile cutoffs (i.e., 0-1)
    clip_rescaled_log_img = np.clip(a=rescaled_log_img, a_min=0, a_max=1)

    # add channel subplot to both figures
    ax_orig = fig_orig.add_subplot(grid_dims[0], grid_dims[1], e + 1)
    ax_log = fig_log.add_subplot(grid_dims[0], grid_dims[1], e + 1)
    ax_clip = fig_clip.add_subplot(grid_dims[0], grid_dims[1], e + 1)

    # plot original channel histogram
    vals, bins, patches = ax_orig.hist(
        img.ravel(), bins=60, color='tab:blue', alpha=0.7, rwidth=0.85
        )
    ax_orig.title.set_text(marker)

    # plot log-transformed channel histogram
    vals, bins, patches = ax_log.hist(
        log_img.ravel(), bins=60, color='tab:blue', alpha=0.7, rwidth=0.85
        )
    ax_log.vlines(
        x=[np.percentile(log_img.ravel(), 0.17),
           np.percentile(log_img.ravel(), 99.99)],
        ymin=0, ymax=vals.max(), color='tab:red'
           )
    ax_log.title.set_text(marker)

    # plot new channel histogram
    vals, bins, patches = ax_clip.hist(
        clip_rescaled_log_img.ravel(), bins=60,
        color='tab:blue', alpha=0.7, rwidth=0.85
        )
    ax_clip.title.set_text(marker)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99, hspace=0.2)
plt.tight_layout()
fig_orig.savefig(os.path.join(save_dir, 'log_hists_orig.pdf'))
fig_log.savefig(os.path.join(save_dir, 'log_hists_log.pdf'))
fig_clip.savefig(os.path.join(save_dir, 'log_hists_clip.pdf'))
plt.close('all')

# save cutoffs to disk
with open(os.path.join(save_dir, 'cutoffs.pkl'), 'wb') as handle:
    pickle.dump(cutoffs, handle, protocol=pickle.HIGHEST_PROTOCOL)
