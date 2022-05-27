import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np
from math import ceil
from math import floor

from itertools import product
from natsort import natsorted

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

import matplotlib.gridspec as gridspec

from skimage.color import gray2rgb

import zarr
from lazy_ops import DatasetView

from keras.models import load_model

from sklearn.manifold import TSNE
from umap import UMAP
import hdbscan
from joblib import Memory

import pickle


def transposeZarr(z):
    view = DatasetView(z)
    result = view.lazy_transpose([1, 2, 3, 0])

    return result


def EncodeImgs(X, encoder):

    X_encoded = encoder.predict(X)

    return X_encoded


def reverse_log(channel):

    # undo percentile scale
    channel = (
        (((upper_cutoff_log-lower_cutoff_log)*(channel.ravel()-0)) /
         (1-0)
         ) + lower_cutoff_log).reshape(channel.shape)

    # exponentiate to undo log10-transform
    channel = 10 ** channel

    # rescale 0-1 to uint bounds
    lower = 10**lower_cutoff_log
    upper = 10**upper_cutoff_log
    channel = (channel-lower) / (upper-lower)

    return channel


def DecodeVectors(X_encoded, orig_input_dims, thumb_channels_to_view):

    # map colors onto channels_to_view
    channel_color_dict = {}
    palette = list(plt.cm.get_cmap('tab10').colors)
    for name, color in zip(thumb_channels_to_view, palette):
        channel_color_dict[name] = (channel_dict[name], color)

    # initialize a numpy array to store reconstructed thumbnails
    X_decoded = np.empty(
        shape=(0, orig_input_dims[0], orig_input_dims[1], 3))

    for j in X_encoded:

        z_sample = np.array([j])

        decoded = decoder.predict(z_sample)

        reconstructed_img = decoded.reshape(
            orig_input_dims[0], orig_input_dims[1], orig_input_dims[2])

        # initialize image overlay
        overlay = np.zeros(
            (reconstructed_img.shape[0],
             reconstructed_img.shape[1]))

        overlay = gray2rgb(overlay)

        for name, (ch, color) in channel_color_dict.items():

            channel = reconstructed_img[:, :, ch]

            channel = reverse_log(channel)

            channel = gray2rgb(channel)

            channel = (channel * color)

            overlay += channel

        overlay = overlay.reshape(
            (1, orig_input_dims[0], orig_input_dims[1], 3)
            )

        X_decoded = np.concatenate((X_decoded, overlay), axis=0)

    return X_decoded, channel_color_dict


def ScatterReconstructions(X_decoded, X_encoded_embedded, zoom, ax):

    def imscatter(x, y, ax, imageData, zoom, intensity_multiplier):

        images = []
        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            img = imageData[i] * intensity_multiplier
            image = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(
                image, (x0, y0), xycoords='data', frameon=False)
            images.append(ax.add_artist(ab))

        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    imscatter(
        X_encoded_embedded[:, 0], X_encoded_embedded[:, 1],
        imageData=X_decoded, ax=ax, zoom=zoom, intensity_multiplier=3.0)


def PlotLatentSpace(reconstructions, zoom, X_encoded_embedded, X_decoded, y, channel_color_dict, scatter_point_size, filename):

    fig, ax = plt.subplots(figsize=(10, 10))

    if reconstructions:

        ScatterReconstructions(
            X_decoded=X_decoded, X_encoded_embedded=X_encoded_embedded,
            zoom=zoom, ax=ax
            )

        custom_lines = []
        for name, (ch, color) in channel_color_dict.items():

            custom_lines.append(
                Line2D([0], [0], color=color, lw=6, label=name)
                )

        ax.scatter(
            X_encoded_embedded[:, 0],
            X_encoded_embedded[:, 1],
            c='k', s=0.0, ec='k', lw=0.25, zorder=4)

        plt.legend(
            handles=custom_lines, prop={'size': 11}, labelspacing=0.7,
            bbox_to_anchor=(1.22, 1.0)
            )

        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{filename}.pdf'))
        plt.close('all')

    else:
        num_labels = len(np.unique(y))
        num_colors = plt.cm.get_cmap('tab20').N
        palette_multiplier = ceil(num_labels/num_colors)
        palette = [(0.0, 0.0, 0.0)]
        palette.extend(list(plt.cm.get_cmap('tab20').colors))
        palette = palette * palette_multiplier

        label_color_dict = dict(zip(sorted(np.unique(y)), palette))

        legend_elements = []
        for lbl, color in label_color_dict.items():

            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       label=lbl, markerfacecolor=color,
                       markeredgecolor='k', lw=0.25, markersize=15)
                       )

        plt.scatter(
            X_encoded_embedded[:, 0],
            X_encoded_embedded[:, 1],
            c=[label_color_dict[i] for i in y],
            ec='k', lw=0.25, s=scatter_point_size
            )

        plt.legend(
            handles=legend_elements, labelspacing=0.8,
            bbox_to_anchor=(1.12, 1.0)
            )

        plt.grid(False)
        plt.savefig(os.path.join(save_dir, f'{filename}.pdf'))
        plt.close('all')

        return label_color_dict


def InterpolationGrid(orig_input_dims, grid_size, X_encoded, y, decoder, label_color_dict, channel_color_dict, frac_of_scatter_points, scatter_point_size, make_sample_sizes_equal, img_brightness_multiplier, scatter_point_alpha):

    # make lists to store grid coordinates and their indices
    # for every latent space dimension
    grids = []
    indices = []

    # grab dimensions in reverse order (e.g. 3, 2, 1, 0) with grids.reverse()
    for d in range(latent_dim):

        # round minimum latent variable in dimension 'd' down to 100th place
        flr = floor(X_encoded[:, d].min() * 100.0) / 100.0

        # round maximum latent variable in dimension 'd' up to 100th place
        cel = ceil(X_encoded[:, d].max() * 100.0) / 100.0

        # construct grid of latent dimension values and their grid indices
        grid = np.array(np.linspace(flr, cel, grid_size))
        grids.append(grid)

        idx = np.array(range(0, len(grid)))
        idx = idx.astype(int)
        indices.append(idx)

    grids.reverse()

    # create an empty array that will fit the required number of
    # thumbnails given the chosen grid size
    y_dim, x_dim, channels = (orig_input_dims[0], orig_input_dims[1],
                              orig_input_dims[2]
                              )
    figure = np.zeros((y_dim * grid_size, x_dim * grid_size))

    # convert to RGB
    figure = gray2rgb(figure)

    # sample the grid
    for grid_tup, dim_tup in zip(product(*grids), product(*indices)):

        grid_tup = list(grid_tup)
        grid_tup.reverse()

        # get z sample at current grid spec
        z_sample = np.array([grid_tup])

        # decode z sample
        X_decoded = decoder.predict(z_sample)

        # reconstruct image
        reconstructed_img = X_decoded.reshape(
            y_dim, x_dim, channels)

        # create blank image to append channels to
        overlay = np.zeros(
            (reconstructed_img.shape[0],
             reconstructed_img.shape[1]))

        # convert to RGB
        overlay = gray2rgb(overlay)

        # append chanenels
        for name, (ch, color) in channel_color_dict.items():

            channel = reconstructed_img[:, :, ch]

            channel = gray2rgb(channel)

            channel = channel * img_brightness_multiplier

            overlay += channel * color

        figure[dim_tup[0] * y_dim: (dim_tup[0] + 1) * y_dim,
               dim_tup[1] * x_dim: (dim_tup[1] + 1) * x_dim] = overlay

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(figure)
    plt.grid(linestyle='dotted', linewidth=0.0)
    plt.gca().invert_yaxis()

    ax.set_xticks(list(range(x_dim, grid_size * x_dim+1, x_dim)))
    ax.set_xticklabels(list(np.round(grids[1], 2)), size=8)
    ax.set_yticks(list(range(y_dim, grid_size * y_dim+1, y_dim)))
    ax.set_yticklabels(list(np.round(grids[0], 2)), size=8)

    y = y.reset_index(drop=True)  # ensure y and X_encoded indices match
    y = y.sample(frac=frac_of_scatter_points)  # sample y
    data = X_encoded[y.index]  # get corresponding X_encoded samples
    y = y.reset_index(drop=True)  # reset y index to matche sampled X_encoded

    scatter_df = pd.concat([pd.DataFrame(y), pd.DataFrame(data)], axis=1)

    if make_sample_sizes_equal is True:
        lengths_list = []
        for i in scatter_df['cluster'].unique():

            lengths_list.append(len(scatter_df[scatter_df['cluster'] == i]))

        sample_size = min(lengths_list)

        sample_dfs = []
        for j in scatter_df['cluster'].unique():
            if len(scatter_df[scatter_df['cluster'] == j]) != sample_size:
                sample_dfs.append(
                    scatter_df[scatter_df['cluster'] == j].sample(n=sample_size))
            else:
                sample_dfs.append(scatter_df[scatter_df['cluster'] == j])

        scatter_df = pd.concat(sample_dfs, axis=0)
    else:
        pass

    # get global x and y ranges
    global_x_min = scatter_df[0].min()
    global_x_max = scatter_df[0].max()
    global_y_min = scatter_df[1].min()
    global_y_max = scatter_df[1].max()

    data = data[scatter_df.index]
    scatter_df = scatter_df.reset_index(drop=True)

    # filter latent vectors to isolate those between
    # the latent variable ranges used to generate the sweep grid
    scatter_points = scatter_df[
        (scatter_df[0] > grids[1].min())
        & (scatter_df[0] < grids[1].max())
        & (scatter_df[1] > grids[0].min())
        & (scatter_df[1] < grids[0].max())
        ].copy()

    # map latent space units (percent point function)
    # to the x, y pixel ranges of the sweep grid
    scatter_points[0] = np.interp(
        scatter_points[0],
        (global_x_min, global_x_max),
        (orig_input_dims[0]/2, (figure.shape[0] - orig_input_dims[0]/2))
        )
    scatter_points[1] = np.interp(
        scatter_points[1],
        (global_y_min, global_y_max),
        (orig_input_dims[0]/2, (figure.shape[0] - orig_input_dims[0]/2))
        )

    # plot latent vectors for images
    for i in natsorted(scatter_points['cluster'].unique()):

        ax.scatter(
            scatter_points[0][scatter_points['cluster'] == i],
            scatter_points[1][scatter_points['cluster'] == i],
            fc=[
                label_color_dict[i] for i in scatter_points['cluster'][
                    scatter_points['cluster'] == i]],
            marker='o',
            label=i, s=scatter_point_size,
            ec='k', lw=0.25, alpha=scatter_point_alpha)

    # channel legend
    legend_elements = []
    for name, (ch, color) in channel_color_dict.items():

        legend_elements.append(Line2D([0], [0], lw=6, color=color, label=name))

    leg = plt.legend(
        handles=legend_elements, loc='upper left', prop={'size': 11},
        markerscale=1, labelspacing=0.7, bbox_to_anchor=(1, 0.32)
        )
    ax.add_artist(leg)

    # cluster legend
    ax.legend(
        loc='upper left', prop={'size': 11}, labelspacing=0.6,
        markerscale=3, bbox_to_anchor=(1, 1.0))

    plt.xticks(rotation=90)

    plt.xlabel(
        'latent dimension 1', size=13,
        labelpad=10, fontweight='normal')
    plt.ylabel(
        'latent dimension 2', size=13,
        labelpad=10, fontweight='normal')

    plt.savefig(
        os.path.join(save_dir, 'InterpolationGrid.pdf'), bbox_inches='tight')
    plt.close('all')

    return global_x_min, global_x_max, global_y_min, global_y_max, scatter_df


class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def LassoVectors(orig_input_dims, imgs_instead_of_points, zoom, X, X_encoded, X_encoded_embedded, X_decoded, y, numColumns, intensity_multiplier, max_examples, label_color_dict, channel_color_dict, thumbnail_font_size):

    lasso_dict = {}

    data = pd.DataFrame(X_encoded_embedded, columns=['x', 'y'])

    if all(y == clustering.labels_):
        y_df = pd.DataFrame(clustering.labels_)
        y_df.rename(columns={0: 'cluster'}, inplace=True)
    else:
        y_df = y

    data = pd.merge(
        y_df.reset_index(drop=True), data, left_index=True, right_index=True
        )

    input_imgs = X[data.index]

    subplot_kw = dict(
        xlim=(data['x'].min(), data['x'].max()),
        ylim=(data['y'].min(), data['y'].max()),
        autoscale_on=False
        )

    fig, lasso_ax = plt.subplots(subplot_kw=subplot_kw, figsize=(9, 8))

    if imgs_instead_of_points is True:

        X_decoded = X_decoded[data.index]
        X_encoded_embedded = X_encoded_embedded[data.index]

        ScatterReconstructions(
            X_decoded=X_decoded, X_encoded_embedded=X_encoded_embedded,
            zoom=zoom, ax=lasso_ax
            )

        legend_elements = []
        for name, (ch, color) in channel_color_dict.items():

            legend_elements.append(
                Line2D([0], [0], color=color, lw=5, label=name)
                )

        pts = lasso_ax.scatter(
            data['x'], data['y'], c='k', s=0.0, ec='k', lw=0.25, zorder=4
            )

        plt.legend(
            handles=legend_elements, markerscale=1, labelspacing=0.7,
            prop={'size': 11}, bbox_to_anchor=(1.02, 0.99)
            )

    else:
        if all(y == clustering.labels_):

            num_labels = len(np.unique(y))
            num_colors = plt.cm.get_cmap('tab20').N
            palette_multiplier = ceil(num_labels/num_colors)
            palette = [(0.0, 0.0, 0.0)]
            palette.extend(list(plt.cm.get_cmap('tab20').colors))
            palette = palette * palette_multiplier

            label_color_dict = dict(zip(sorted(np.unique(y)), palette))

            c = [
                'k' if i == -1 else label_color_dict[i] for
                i in clustering.labels_
                ]

            legend_elements = []
            for i in np.unique(clustering.labels_):
                if i == -1:
                    markerfacecolor = 'k'
                else:
                    markerfacecolor = label_color_dict[i]

                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w',
                           label=i, markerfacecolor=markerfacecolor,
                           markeredgecolor='k', lw=0.25, markersize=15)
                    )
        else:

            c = [label_color_dict[i] for i in data['cluster']]

            legend_elements = []
            for name, color in label_color_dict.items():

                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w',
                           label=name, markerfacecolor=color,
                           markeredgecolor='k', lw=0.25, markersize=15)
                    )

        pts = lasso_ax.scatter(
            data['x'], data['y'], c=c, s=30.0, ec='k', lw=0.25, zorder=4
            )

        lasso_ax.update_datalim(np.column_stack([data['x'], data['y']]))
        lasso_ax.autoscale()

        plt.legend(
            handles=legend_elements, markerscale=1, labelspacing=0.7,
            prop={'size': 11}, bbox_to_anchor=(1.02, 0.99)
            )

    selector = SelectFromCollection(lasso_ax, pts)

    latent_vectors = X_encoded[data.index]
    data = data.reset_index(drop=True)

    def accept(event):
        if event.key == "enter":
            print("Selected points:")
            print(selector.xys[selector.ind])
            selector.disconnect()
            lasso_ax.set_title("")
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    lasso_ax.set_title("Press enter to accept selected points.")
    lasso_ax.set_aspect('equal')
    plt.show(block=True)

    selected_vectors = data.loc[selector.ind]
    selected_vectors['latent_vector'] = [
        i for i in latent_vectors[selector.ind]]
    selected_vectors['input_img'] = [
        i.flatten() for i in input_imgs[selector.ind]]

    if max_examples is not None:
        if len(selected_vectors) < max_examples:
            max_examples = len(selected_vectors)
        selected_vectors = selected_vectors.sample(
            n=max_examples, random_state=44)

    selected_vectors.sort_values(by='cluster', inplace=True)

    # check cell images
    numSamples = len(selected_vectors)
    numRows = ceil(numSamples/numColumns)
    grid_dims = (numRows, numColumns)

    fig = plt.figure()

    fig.text(0.13, 0.97, 'Input Images', ha='left', fontsize='medium')
    fig.text(0.53, 0.97, 'Learned Representations', fontsize='medium')

    outer_grid_rows = 1
    outer_grid_cols = 2

    outer = gridspec.GridSpec(
        outer_grid_rows, outer_grid_cols, wspace=0.1, hspace=0.0)

    for panel in range(outer_grid_rows * outer_grid_cols):

        inner = gridspec.GridSpecFromSubplotSpec(
            grid_dims[0], grid_dims[1],
            subplot_spec=outer[panel], wspace=0.1, hspace=0.0)

        for e, row in enumerate(selected_vectors.iterrows()):

            ax = plt.Subplot(fig, inner[e])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            if panel == 0:

                input_img = row[1]['input_img'].reshape(
                    orig_input_dims[0], orig_input_dims[1], orig_input_dims[2]
                    )

                overlay = np.zeros(
                    (input_img.shape[0],
                     input_img.shape[1]))

                overlay = gray2rgb(overlay)

                for name, (ch, color) in channel_color_dict.items():

                    channel_slice = input_img[:, :, ch]

                    channel_slice = reverse_log(channel_slice)

                    channel_slice = gray2rgb(channel_slice)

                    channel_slice = channel_slice * intensity_multiplier

                    overlay += channel_slice * color

            elif panel == 1:

                z_sample = np.array([list(row[1]['latent_vector'])])

                X_decoded = decoder.predict(z_sample)

                reconstructed_img = X_decoded.reshape(
                    orig_input_dims[0], orig_input_dims[1], orig_input_dims[2])

                overlay = np.zeros(
                    (reconstructed_img.shape[0],
                     reconstructed_img.shape[1])
                     )

                overlay = gray2rgb(overlay)

                for name, (ch, color) in channel_color_dict.items():

                    channel_slice = reconstructed_img[:, :, ch]

                    channel_slice = reverse_log(channel_slice)

                    channel_slice = gray2rgb(channel_slice)

                    channel_slice = channel_slice * intensity_multiplier

                    overlay += channel_slice * color

            ax.imshow(overlay, cmap=plt.cm.binary)

            ax.set_xlabel(
                row[1]['cluster'], fontsize=thumbnail_font_size, labelpad=0.75
                )
            fig.add_subplot(ax)

    fig.subplots_adjust(
        bottom=0.01, top=0.94,
        left=0.01, right=0.85,
        wspace=0.2, hspace=0.1
        )

    legend_elements = []
    for name, (ch, color) in channel_color_dict.items():
        legend_elements.append(
            Line2D([0], [0], color=color, lw=3, label=name)
            )

    fig.legend(
        handles=legend_elements, prop={'size': 5}, bbox_to_anchor=(0.98, 0.95))

    plt.savefig(
        os.path.join(save_dir, 'lassoed_cells.pdf'), bbox_inches='tight'
        )
    plt.close('all')


def PlotReconstructedImages(orig_input_dims, X, X_encoded, y, numColumns, label_color_dict, channel_color_dict, intensity_multiplier, thumbnail_font_size, filename):

    numSamples = len(X)
    numRows = ceil(numSamples/numColumns)
    grid_dims = (numRows, numColumns)

    fig = plt.figure()

    fig.text(0.13, 0.97, 'Input Images', ha='left', fontsize='medium')
    fig.text(0.53, 0.97, 'Learned Representations', fontsize='medium')

    outer_grid_rows = 1
    outer_grid_cols = 2

    outer = gridspec.GridSpec(
        outer_grid_rows, outer_grid_cols, wspace=0.1, hspace=0.0
        )

    for panel in range(outer_grid_rows * outer_grid_cols):

        inner = gridspec.GridSpecFromSubplotSpec(
            grid_dims[0], grid_dims[1],
            subplot_spec=outer[panel], wspace=0.1, hspace=0.0)

        for e, (i, j, k) in enumerate(zip(X, X_encoded, y.iteritems())):

            ax = plt.Subplot(fig, inner[e])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            if panel == 0:

                overlay = np.zeros((i.shape[0], i.shape[1]))

                overlay = gray2rgb(overlay)

                for name, (ch, color) in channel_color_dict.items():

                    channel_slice = i[:, :, ch]

                    channel_slice = reverse_log(channel_slice)

                    channel_slice = gray2rgb(channel_slice)

                    channel_slice = channel_slice * intensity_multiplier

                    overlay += channel_slice * color

            elif panel == 1:

                z_sample = np.array([j])

                x_decoded = decoder.predict(z_sample)

                reconstructed_img = x_decoded.reshape(
                    orig_input_dims[0], orig_input_dims[1], orig_input_dims[2])

                overlay = np.zeros(
                    (reconstructed_img.shape[0],
                     reconstructed_img.shape[1])
                     )

                overlay = gray2rgb(overlay)

                for name, (ch, color) in channel_color_dict.items():

                    channel_slice = reconstructed_img[:, :, ch]

                    channel_slice = reverse_log(channel_slice)

                    channel_slice = gray2rgb(channel_slice)

                    channel_slice = channel_slice * intensity_multiplier

                    overlay += channel_slice * color

            ax.imshow(overlay, cmap=plt.cm.binary)

            ax.set_xlabel(
                k[1], fontsize=thumbnail_font_size, labelpad=0.75
                )
            fig.add_subplot(ax)

    fig.subplots_adjust(
        bottom=0.01, top=0.94,
        left=0.01, right=0.85,
        wspace=0.2, hspace=0.1
        )

    legend_elements = []
    for name, (ch, color) in channel_color_dict.items():
        legend_elements.append(
            Line2D([0], [0], color=color, lw=3, label=name)
            )

    fig.legend(
        handles=legend_elements, prop={'size': 5}, bbox_to_anchor=(0.98, 0.95))

    plt.savefig(os.path.join(save_dir, f'{filename}.pdf'), bbox_inches='tight')
    plt.close('all')


def mse(orig_input_dims, X, X_encoded, y, mse_percentile_cutoff, filename):

    errors = []

    for input_img, encoded_img in zip(X, X_encoded):

        z_sample = np.array([encoded_img])

        x_decoded = decoder.predict(z_sample)

        reconstructed_img = x_decoded.reshape(
            orig_input_dims[0], orig_input_dims[1], orig_input_dims[2])

        err = np.sum((input_img - reconstructed_img) ** 2)
        err /= float(input_img.shape[0] * input_img.shape[1])

        errors.append(err)

    average_error = np.mean(errors)
    print(f'average mean squared error is {average_error}')

    n, bins, pathes = plt.hist(errors, bins=50)
    plt.axvline(np.percentile(errors, mse_percentile_cutoff), c='r')
    plt.savefig(os.path.join(save_dir, f'{filename}.pdf'))

    outlier_idxs = [
        i for i, v in enumerate(errors)
        if v > np.percentile(errors, mse_percentile_cutoff)]

    X_outliers = X[outlier_idxs]
    X_encoded_outliers = X_encoded[outlier_idxs]
    y_outliers = y[outlier_idxs].reset_index(drop=True)

    plt.close('all')

    return average_error, errors, X_outliers, X_encoded_outliers, y_outliers, outlier_idxs


# def PlotInputImgs(orig_input_dims, numExamples, numColumns, X, y, channel_color_dict, thumbnail_font_size, filename):
#
#     numSamples = len(X)
#     numRows = ceil(numSamples/numColumns)
#     grid_dims = (numRows, numColumns)
#
#     # numColumns = math.ceil(numExamples/numRows)
#     # grid_dims = (numRows, numColumns)
#
#     ordered_lbs = natsorted(y)
#     # ordered_lbs = y
#
#     # sns.set_style('whitegrid')
#     fig = plt.figure(figsize=(20, 10))
#
#     for e, i in enumerate(ordered_lbs):
#
#         plt.subplot(grid_dims[0], grid_dims[1], e + 1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#
#         # initialize array of zeros with shape of full-size image
#         overlay = np.zeros((orig_input_dims[0], orig_input_dims[1]))
#         overlay = gray2rgb(overlay)
#
#         for name, (ch, color) in channel_color_dict.items():
#             channel_slice = X[e, :, :, ch]
#             channel_slice = gray2rgb(channel_slice)
#             channel_slice = channel_slice * color
#             overlay += channel_slice
#         plt.imshow(overlay, cmap=plt.cm.binary)
#         plt.xlabel(i, size=thumbnail_font_size, labelpad=3.0)
#
#     legend_elements = []
#     for name, (ch, color) in channel_color_dict.items():
#         legend_elements.append(Line2D([0], [0], color=color, lw=5, label=name))
#
#     fig.legend(
#         handles=legend_elements, prop={'size': 11}, bbox_to_anchor=(0.97, 0.89)
#         )
#
#     plt.savefig(os.path.join(save_dir, f'{filename}.pdf'))
#     plt.close('all')

latent_dim = 850  # should be the same as that used for training
training_thumb_dims = (30, 30, 21)
embedding_algorithm = 'UMAP'  # 'TSNE'

# read percentile cutoffs selected in script 3_feature_preprocessing
with open(
  '/Users/greg/projects/vae_sardana-097/4_feature_preprocessing_selections'
  '/cutoffs.pkl',
  'rb') as handle:
    cutoffs = pickle.load(handle)

# complete list of ordered channels in thumbnail data
markers = [
    'anti_CD3', 'anti_CD45RO', 'Keratin_570', 'aSMA_660', 'CD4_488',
    'CD45_PE', 'PD1_647', 'CD20_488', 'CD68_555', 'CD8a_660', 'CD163_488',
    'FOXP3_570', 'PDL1_647', 'Ecad_488', 'Vimentin_555', 'CDX2_647',
    'LaminABC_488', 'Desmin_555', 'CD31_647', 'PCNA_488', 'CollagenIV_647'
    ]
channel_dict = dict(zip(markers, range(len(markers))))

thumb_channels_to_view = [
    'Keratin_570', 'CD20_488', 'aSMA_660', 'CD4_488', 'CD8a_660',
    'FOXP3_570', 'PCNA_488', 'CD68_555', 'Ecad_488'
    ]
thumb_channels_to_view = [
    'Keratin_570', 'CD8a_660', 'CD4_488', 'CD163_488', 'Vimentin_555',
    ]

###############################################################################

# save directory
save_dir = (
    f'/Users/greg/projects/vae_sardana-097/6_latent_space_LD{latent_dim}/'
    )
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

###############################################################################

# load previously saved encoder and decoders
try:
    encoder = load_model(
        '/Users/greg/projects/vae_sardana-097/5_train_vae/encoder.hdf5'
        )
except OSError:
    print('Encoder not found.')
    sys.exit()

try:
    decoder = load_model(
        '/Users/greg/projects/vae_sardana-097/5_train_vae/decoder.hdf5'
        )
except OSError:
    print('Decoder not found.')
    sys.exit()

###############################################################################

# read test labels
y_test = pd.read_csv(
    '/Users/greg/projects/vae_sardana-097/1_cellcutter_input/test.csv'
    )

# read floating point test thumbnails
z1_test_path = (
    '/Users/greg/projects/vae_sardana-097/2_cellcutter_output_win30/' +
    'test_thumbnails_30'
    )
X_test = zarr.open(z1_test_path)

# take a sample of thumbnail data
# X_test1 = X_test[:, 0:47991, :, :]
# y_test1 = y_test[0:47991]

X_test1 = X_test[:, 0:5000, :, :]
y_test1 = y_test[0:5000]

# convert back to Zarr format after slicing
z = zarr.zeros(
    shape=(X_test1.shape[0], X_test1.shape[1],
           X_test1.shape[2], X_test1.shape[3]),
    chunks=(X_test.chunks[0], X_test.chunks[1],
            X_test.chunks[2], X_test.chunks[3]),
    compressor=X_test.compressor,
    dtype='float32'
            )
z[:] = X_test1

# rearrange Zarr dimensions to fit shape of expected VAE input
# (i.e. cells, x, y, channels)
X_test1 = transposeZarr(z=z)

# load data into memory
X_test1 = X_test1[:]

# read percentile cutoffs selected in script 3_feature_preprocessing
with open(
  '/Users/greg/projects/vae_sardana-097/4_feature_preprocessing_selections'
  '/cutoffs.pkl',
  'rb') as handle:
    cutoffs = pickle.load(handle)

# log10 transform
X_test1 = np.log10(X_test1, where=(X_test1 != 0))

for i in range(X_test1.shape[0]):

    for e, (lower_cutoff_log, upper_cutoff_log) in enumerate(
      cutoffs.values()):

        # scale 0.17th and 99.99th percentile between 0 and 1
        X_test1[i, :, :, e] = (
            (((1-0)*(X_test1[i, :, :, e].ravel()-lower_cutoff_log)) /
             (upper_cutoff_log-lower_cutoff_log)
             ) + 0).reshape(X_test1[i, :, :, e].shape)

        # clip lower and upper outliers to 0 and 1, respectively
        X_test1[i, :, :, e] = np.clip(
            a=X_test1[i, :, :, e], a_min=0, a_max=1
            )
###############################################################################

# encode test images
X_encoded = EncodeImgs(X=X_test1, encoder=encoder)

###############################################################################
# embed latent vectors if they are greater than 2D

embedding_path = (
    '/Users/greg/projects/vae_sardana-097/6_latent_space/embedding.npy'
    )

if (latent_dim > 2) and not os.path.exists(embedding_path):

    startTime = datetime.now()

    if embedding_algorithm == 'TSNE':
        print('Computing TSNE embedding.')
        X_encoded_embedded = TSNE(
            n_components=2,
            perplexity=27,
            early_exaggeration=19,
            learning_rate=200.0,
            metric='euclidean',
            random_state=5,
            init='pca', n_jobs=-1).fit_transform(X_encoded)

    elif embedding_algorithm == 'UMAP':
        print('Computing UMAP embedding.')
        X_encoded_embedded = UMAP(
            n_components=2,
            n_neighbors=6,
            learning_rate=1.0,
            output_metric='euclidean',
            min_dist=0.1,
            repulsion_strength=7,
            random_state=4,
            n_epochs=1000,
            init='spectral',
            metric='euclidean',
            metric_kwds=None,
            output_metric_kwds=None,
            n_jobs=-1,
            low_memory=False,
            spread=1.0,
            local_connectivity=1.0,
            set_op_mix_ratio=0.5,
            negative_sample_rate=5,
            transform_queue_size=4.0,
            a=None,
            b=None,
            angular_rp_forest=False,
            target_n_neighbors=-1,
            target_metric='categorical',
            target_metric_kwds=None,
            target_weight=0.5,
            transform_seed=42,
            transform_mode='embedding',
            force_approximation_algorithm=False,
            verbose=False,
            unique=False,
            densmap=False,
            dens_lambda=2.0,
            dens_frac=0.6,
            dens_var_shift=0.1,
            disconnection_distance=None,
            output_dens=False).fit_transform(X_encoded)

    print('Embedding completed in ' + str(datetime.now() - startTime))

    # save embedding
    np.save(os.path.join(save_dir, 'embedding'), X_encoded_embedded)

elif (latent_dim > 2) and os.path.exists(embedding_path):

    # load previously saved embedding
    X_encoded_embedded = np.load(embedding_path)

else:
    # simply assign the 2D X_encoded the variable X_encoded_embedded
    X_encoded_embedded = X_encoded.copy()

###############################################################################

# cluster the data with HDBSCAN
for i in range(285, 286, 1):

    print(f'Minimum_cluster_size is {i}')

    clustering = hdbscan.HDBSCAN(
        min_cluster_size=i, min_samples=None,
        metric='euclidean', alpha=1.0, p=None, algorithm='best',
        leaf_size=40,
        memory=Memory(location=None),
        approx_min_span_tree=True,
        gen_min_span_tree=False, core_dist_n_jobs=4,
        cluster_selection_method='eom',
        allow_single_cluster=False,
        prediction_data=False,
        match_reference_implementation=False).fit(X_encoded_embedded)

    print(np.unique(clustering.labels_))

###############################################################################

# plot latent vectors colored according to prior clustering
label_color_dict = PlotLatentSpace(
    reconstructions=False,
    zoom=None,
    X_encoded_embedded=X_encoded_embedded,
    X_decoded=None,
    y=y_test1['cluster'],
    channel_color_dict=None,
    scatter_point_size=30,
    filename='consensus_clustering'
    )

# plot latent vectors colored according to HDBSCAN clustering of latent space
PlotLatentSpace(
    reconstructions=False,
    zoom=None,
    X_encoded_embedded=X_encoded_embedded,
    X_decoded=None,
    y=clustering.labels_,
    channel_color_dict=None,
    scatter_point_size=30,
    filename='latent_clustering'
    )

# reconstruct thumbnail images from latent vectors
X_decoded, channel_color_dict = DecodeVectors(
    X_encoded=X_encoded, orig_input_dims=training_thumb_dims,
    thumb_channels_to_view=thumb_channels_to_view
    )

# plot latent vectors represented as their learned reconstructions
PlotLatentSpace(
    reconstructions=True,
    zoom=0.5,
    X_encoded_embedded=X_encoded_embedded,
    X_decoded=X_decoded,
    y=y_test1['cluster'],
    channel_color_dict=channel_color_dict,
    scatter_point_size=30,
    filename='thumbnails'
    )

# display learned representations of input thumbnail images
if latent_dim == 2:

    InterpolationGrid(
        orig_input_dims=training_thumb_dims,
        grid_size=50,
        X_encoded=X_encoded,
        y=y_test1['cluster'],
        decoder=decoder,
        label_color_dict=label_color_dict,
        channel_color_dict=channel_color_dict,
        frac_of_scatter_points=1.0,
        scatter_point_size=30.0,
        make_sample_sizes_equal=False,
        img_brightness_multiplier=1.2,
        scatter_point_alpha=1.0,
        )

# get input and output images of lassoed latent vectors
LassoVectors(
    orig_input_dims=training_thumb_dims,
    imgs_instead_of_points=True,
    zoom=0.5,
    X=X_test1,
    X_encoded=X_encoded,
    X_encoded_embedded=X_encoded_embedded,
    X_decoded=X_decoded,
    y=y_test1['cluster'],
    numColumns=10,
    intensity_multiplier=3.0,
    label_color_dict=label_color_dict,
    channel_color_dict=channel_color_dict,
    max_examples=1000,
    thumbnail_font_size=3.0
    )

PlotReconstructedImages(
    orig_input_dims=training_thumb_dims,
    X=X_test1[0:100],
    X_encoded=X_encoded[0:100],
    y=y_test1['cluster'][0:100],
    numColumns=10,
    label_color_dict=label_color_dict,
    channel_color_dict=channel_color_dict,
    intensity_multiplier=3.0,
    thumbnail_font_size=3.0,
    filename='learned_reconstructions'
    )

# compute mean squared error between thumbnail image inputs and outputs
(average_error,
    errors,
    X_outliers,
    X_encoded_outliers,
    y_outliers,
    outlier_idxs) = mse(
        orig_input_dims=training_thumb_dims,
        X=X_test1,
        X_encoded=X_encoded,
        y=y_test1['cluster'],
        mse_percentile_cutoff=99,
        filename='mse_dist'
        )

# get input thumbnails associated with poor learned reconstruction
PlotReconstructedImages(
    orig_input_dims=training_thumb_dims,
    X=X_outliers,
    X_encoded=X_encoded_outliers,
    y=y_outliers,
    numColumns=10,
    label_color_dict=label_color_dict,
    channel_color_dict=channel_color_dict,
    intensity_multiplier=3.0,
    thumbnail_font_size=3.0,
    filename='outliers'
    )
