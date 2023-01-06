import os
import yaml

import pandas as pd
import numpy as np

import math

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from skimage.color import gray2rgb
from skimage.util import img_as_float

import zarr

from tifffile import imread


def PlotInputImgs(numExamples, numColumns, imgs, labels, fontSize, colors, channelNames, channelIDs, fileName):

    numSamples = len(imgs)
    numRows = math.ceil(numExamples/numColumns)
    grid_dims = (numRows, numColumns)

    # numColumns = math.ceil(numExamples/numRows)
    # grid_dims = (numRows, numColumns)

    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(13, 10))

    custom_lines = []
    for e, row in enumerate(labels.iterrows()):

        plt.subplot(grid_dims[0], grid_dims[1], e + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # initialize array of zeros with shape of full-size image
        overlay = np.zeros((imgs.shape[2], imgs.shape[3]))

        overlay = gray2rgb(overlay)

        for d, ch, color in zip(channelIDs, channelNames, colors):

            lyr = imgs[d, row[0], :, :]

            lyr = img_as_float(lyr)

            # apply image contrast settings
            lyr -= (contrast_limits[ch][0]/65535)
            lyr /= (
                (contrast_limits[ch][1]/65535)
                - (contrast_limits[ch][0]/65535))

            lyr = np.clip(lyr, 0, 1)

            lyr = gray2rgb(lyr)
            lyr = lyr * color
            overlay += lyr

            custom_lines.append(Line2D([0], [0], color=color, lw=5))

        label = row[1]['cluster']

        # pass "thumb" variable to imshow (below) to check whether
        # thumbnail zarr file and CSV indices line up

        # img = imread(
        #     '/Volumes/My Book/cylinter_input/sardana-097/tif/' +
        #     'WD-76845-097.ome.tif', key=0)
        #
        # thumb = img[
        #     round(labels.loc[row[0], "Y_centroid"]) -
        #     32:round(labels.loc[row[0], "Y_centroid"]) + 32,
        #     round(labels.loc[row[0], "X_centroid"]) -
        #     32:round(labels.loc[row[0], "X_centroid"]) + 32,
        #     ]

        plt.imshow(overlay, cmap=plt.cm.binary)
        plt.xlabel(label, size=fontSize, labelpad=1.5)

    fig.legend(
        custom_lines, channelNames, prop={'size': 11},
        bbox_to_anchor=(0.98, 0.99)
        )

    plt.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.85)
    plt.savefig(os.path.join(save_dir, f'{fileName}.pdf'))
    plt.close('all')


###############################################################################

# specific number of thumbnails to view
num_examples = 240

###############################################################################

# specific channels/ids/colors

# channel_names = [
#     'anti_CD3', 'anti_CD45RO', 'Keratin_570', 'aSMA_660',
#     'CD4_488', 'CD45_PE', 'PD1_647', 'CD20_488', 'CD68_555', 'CD8a_660',
#     'CD163_488', 'FOXP3_570', 'PDL1_647', 'Ecad_488', 'Vimentin_555',
#     'CDX2_647', 'LaminABC_488', 'Desmin_555', 'CD31_647', 'PCNA_488',
#     'CollagenIV_647'
#     ]

channel_names = [
    'Ecad_488', 'Keratin_570', 'aSMA_660', 'CD4_488',
    'CD20_488', 'CD8a_660', 'FOXP3_570', 'Vimentin_555'
    ]

channel_ids = [13, 2, 3, 4, 7, 9, 11, 14]

###############################################################################

# save directory
save_dir = ('/Users/greg/projects/vae/output/3_thumbnail_examples')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

###############################################################################

# read training labels
labels = pd.read_csv(
    '/Users/greg/projects/vae/output/1_cellcutter_input/train.csv'
    )

# read training images
z_path = (
    '/Users/greg/projects/vae/output/2_cellcutter_output_win30/' +
    'train_thumbnails_30'
    )
z = zarr.open(z_path, mode='r')

###############################################################################

# contrast settings
contrast_path = (
    '/Volumes/My Book/cylinter_input/clean_quant/output_3d_v2/contrast/' +
    'contrast_limits.yml'
    )

contrast_limits = yaml.safe_load(open(contrast_path))

###############################################################################

# pull random thumbnails from training data to check quality
thumb_ids = np.random.RandomState(1).choice(
    range(0, z.shape[1]), num_examples, replace=False)

imgs = z.get_orthogonal_selection((slice(None), thumb_ids))

labels = labels.iloc[thumb_ids]
labels.reset_index(drop=True, inplace=True)
labels.sort_values(by='cluster', inplace=True)

colors = plt.get_cmap('tab10').colors * math.ceil(imgs.shape[3]/10)

PlotInputImgs(
    numExamples=num_examples,
    numColumns=16,
    imgs=imgs,
    labels=labels,
    fontSize=8,
    colors=colors,
    channelNames=channel_names,
    channelIDs=channel_ids,
    fileName='thumbnail_examples'
    )
