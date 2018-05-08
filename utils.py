from cloud_run import cloud_run

# cloud run?
if cloud_run():
    import matplotlib as mpl
    mpl.use('Agg')

import os
import shutil
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# data partition directory
data_partition_dir = os.path.join('DataPartitions')


def get_image_files():

    # get data directory contents
    data_dir = os.path.join(os.getcwd(), 'Data')
    listing = os.listdir(data_dir)

    # loop over subject files
    image_files = []
    for s in listing:

        # subject directory
        sub_dir = os.path.join(data_dir, s)

        # is this a directory?
        if os.path.isdir(sub_dir):

            # get all tiff files
            tiff_files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith('.tif')]

            # extend list
            image_files.extend(tiff_files)

    return image_files


def partition_image(image_file, partition_dim, overlap):

    # compute move distance
    mv_dist = partition_dim - overlap

    # load data into np array (excluding grey scale inverse composite)
    im = io.imread(image_file)[:-1]
    im = np.transpose(im, [1, 2, 0])

    # compute number of partitions
    n_row_parts = int(im.shape[0] / mv_dist)
    n_col_parts = int(im.shape[1] / mv_dist)
    n_parts = n_row_parts * n_col_parts

    # loop over the rows
    idx = 0
    im_parts = np.zeros([n_parts, mv_dist, mv_dist, im.shape[-1]], dtype=np.float64)
    for r in range(0, im.shape[0], mv_dist):

        # loop over the columns
        for c in range(0, im.shape[1], mv_dist):

            # continue if not enough room
            if r + mv_dist > im.shape[0] or c + mv_dist > im.shape[1]:
                continue

            # load the partition
            im_parts[idx] = im[r:r + mv_dist, c:c + mv_dist]
            idx += 1

    return im_parts, n_parts


def stitch_partition(image_file, x_parts, x_hat_parts):

    # load data into np array (excluding grey scale inverse composite)
    im = io.imread(image_file)[:-1]
    im = np.transpose(im, [1, 2, 0])

    # compute number of partitions
    n_row_parts = int(im.shape[0] / x_parts.shape[1])
    n_col_parts = int(im.shape[1] / x_parts.shape[2])

    # loop over the rows
    idx = 0
    im_truth = np.zeros([n_row_parts * x_parts.shape[1], n_col_parts * x_parts.shape[1], im.shape[-1]])
    im_recon = np.zeros([n_row_parts * x_parts.shape[2], n_col_parts * x_parts.shape[2], im.shape[-1]])
    for r in range(0, n_row_parts):

        # loop over the columns
        for c in range(0, n_col_parts):

            # determine insertion indices
            r_start = r * x_parts.shape[1]
            r_stop = (r + 1) * x_parts.shape[1]
            c_start = c * x_parts.shape[2]
            c_stop = (c + 1) * x_parts.shape[2]

            # load the partition
            im_truth[r_start:r_stop, c_start:c_stop] = x_parts[idx]
            im_recon[r_start:r_stop, c_start:c_stop] = x_hat_parts[idx]
            idx += 1

    return im_truth, im_recon


def partition_data(partition_dim=64, overlap=0):

    # make sure overlap is in range and that dimensions are a power of 2 for auto-encoder
    assert 0 <= overlap < partition_dim
    assert np.log2(partition_dim) == int(np.log2(partition_dim))

    # make a fresh directory
    if os.path.exists(data_partition_dir):
        shutil.rmtree(data_partition_dir)
    os.makedirs(data_partition_dir)

    # get image files
    image_files = get_image_files()

    # loop over images
    total_samples = 0
    for i in range(len(image_files)):

        # partition the image
        im_parts, n_parts = partition_image(image_files[i], partition_dim, overlap)

        # tally up number of samples
        total_samples += n_parts

        # save partitioned data for this subject
        np.save(os.path.join(data_partition_dir, 'data_partition_{:d}.npy'.format(i)), im_parts)

        # print update
        update_str = 'Partitioning Data. Percent Complete = {:.2f}%'.format(100 * (i + 1) / len(image_files))
        print('\r' + update_str, end='')

    # print completion message
    print('\nPartitioned data into {:d} image partitions.'.format(total_samples))


def load_partitioned_data(n_files=None):

    # get all npy files
    npy_files = [os.path.join(data_partition_dir, f) for f in os.listdir(data_partition_dir) if f.endswith('.npy')]
    if n_files is not None:
        npy_files = npy_files[:n_files]

    # compute number of samples
    num_samples = 0
    for i in range(len(npy_files)):
        num_samples += np.load(npy_files[i]).shape[0]
        print('\rComputing number of samples... {:.2f}% complete'.format(100 * (i + 1) / len(npy_files)), end='')
    print('\nTotal samples = {:d}'.format(num_samples))

    # get dimensions
    image_dim = list(np.load(npy_files[0]).shape[1:])

    # initialize result
    idx = 0
    data = np.zeros([num_samples] + image_dim)

    # load the data
    for i in range(len(npy_files)):

        # print update
        update_str = 'Loading data. Percent Complete = {:.2f}%'.format(100 * (i + 1) / len(npy_files))
        print('\r' + update_str, end='')

        # load the data
        x = np.load(npy_files[i])
        data[idx:idx + x.shape[0]] = x
        idx += x.shape[0]

    # print completion message
    print('\nLoading data complete!')

    return data


def split_data(n_samps, percent_test):
    """
    :param n_samps: number of data samples
    :param percent_test: percent of data to hold out
    :return: two sets of indices corresponding to training and validation data
    """

    # generate and randomly shuffle
    idx = np.arange(n_samps)
    np.random.shuffle(idx)

    # determine cut-point
    i_cut = int(n_samps * (1 - percent_test))

    # generate train and test indices
    i_train = idx[:i_cut]
    i_valid = idx[i_cut:]

    return i_train, i_valid


def get_batches(data_len, batch_size, shuffle=True):

    # get indices
    indices = np.arange(data_len)

    # shuffle if specified
    if shuffle:
        np.random.shuffle(indices)

    # determine batch list
    batches = []
    while len(indices) > 0:
        batches.append(indices[:batch_size])
        indices = indices[batch_size:]

    # ensure all samples going through
    total_batch_len = sum([len(batch) for batch in batches])
    assert total_batch_len == data_len, 'Missing elements!'

    return batches


def plot(x, x_hat, fig=None, super_title=None, save_loc=None, data='cont'):

    # generate figure if its not supplied
    if fig is None:
        fig = plt.figure()

    # otherwise clear the figure for redrawing
    else:
        fig.clf()

    # compute number of images
    n_channels = x.shape[-1]

    # channel labels
    if n_channels == 7:
        ch_labels = ['GITR',
                     'IDO',
                     'Ki-67',
                     'Foxp3',
                     'CD8',
                     'DAPI',
                     'Cytokeratins']

    # loop over the channels
    for i in range(n_channels):

        # continuous data
        if data == 'cont':

            # scale max/min appropriately
            x_min = np.min([np.min(x[:, :, i]), np.min(x_hat[:, :, i])])
            x_max = 3 * np.max([np.std(x[:, :, i]), np.std(x_hat[:, :, i])])

        # binary data
        else:

            # scale max/min appropriately
            x_min = np.min([np.min(x), np.min(x_hat)])
            x_max = np.max([np.max(x), np.max(x_hat)])

        # generate subplots for original data
        sp = fig.add_subplot(2, n_channels, i + 1)
        sp.imshow(np.flipud(x[:, :, i]), vmin=x_min, vmax=x_max, origin='lower')
        sp.set_xticks([])
        sp.set_yticks([])
        if i == 0:
            sp.set_ylabel('Original')
        if n_channels == 7:
            sp.set_title(ch_labels[i])

        # generate subplots for reconstructed data
        sp = fig.add_subplot(2, n_channels, i + 1 + n_channels)
        sp.imshow(np.flipud(x_hat[:, :, i]), vmin=x_min, vmax=x_max, origin='lower')
        sp.set_xticks([])
        sp.set_yticks([])
        if i == 0:
            sp.set_ylabel('Reconstructed')
        if n_channels == 7:
            sp.set_title(ch_labels[i])

    # make it tight
    plt.tight_layout()

    # super title provided
    if super_title is not None:
        plt.suptitle(super_title)

    # interactive plotting
    if save_loc is None:
        plt.pause(0.05)

    # saving data
    else:
        fig.savefig(os.path.join(save_loc))
        plt.clf()

    return fig


if __name__ == '__main__':

    # get image files
    image_files = get_image_files()

    # partition the data
    partition_data()
