from run_mode import test_mode

# real run?
if not test_mode():
    import matplotlib as mpl
    mpl.use('Agg')

import os
import pickle
import tensorflow as tf
from matplotlib import pyplot as plt
from utils import get_image_files, partition_image, plot, stitch_partition
from models import AutoEncoder, VariationalAutoEncoder, mdl_reconstruction

# result base directory
if not test_mode():
    RESULT_DIR = os.path.join(os.getcwd(), 'ResultsReal')
else:
    RESULT_DIR = os.path.join(os.getcwd(), 'ResultsTest')

# reconstruction directory
RECON_DIR = os.path.join(os.getcwd(), 'Reconstruction')

# result directories
mdl_dirs = [os.path.join('AE', 'Bern', 'Arch1'),
            os.path.join('AE', 'Bern', 'Arch2'),
            os.path.join('AE', 'Gauss', 'Arch1'),
            os.path.join('AE', 'Gauss', 'Arch2'),
            os.path.join('VAE', 'Bern', 'Arch1'),
            os.path.join('VAE', 'Bern', 'Arch2'),
            os.path.join('VAE', 'Gauss', 'Arch1'),
            os.path.join('VAE', 'Gauss', 'Arch2')]


def reconstruct_images(mdl, mdl_params, mdl_dir):

    # get image files
    image_files = get_image_files()

    # begin new session
    with tf.Session() as sess:

        # run initialization
        sess.run(tf.global_variables_initializer())

        # load the model
        mdl.saver.restore(sess, os.path.join(mdl.save_dir, 'model.ckpt'))

        # declare reusable figure handle
        fig = plt.figure(figsize=(25.7, 6.77))

        # loop over images
        for i in range(len(image_files)):

            # partition the image
            im_parts, _ = partition_image(image_files[i], mdl_params['input_dim'][0], 0)

            # generate_reconstruction
            x, x_hat = mdl_reconstruction(sess, mdl, mdl_params, im_parts)

            # stitch the partition
            im_truth, im_recon = stitch_partition(image_files[i], x, x_hat)

            # continuous data
            if mdl_params['px_z'] == 'Gaussian':
                data = 'cont'

            # binary data
            else:
                data = 'binary'

            # plot the figure
            title = mdl_dir.replace(os.sep, ' ')
            save_loc =os.path.join(recon_dir, 'im_{:d}.png'.format(i))
            plot(im_truth, im_recon, fig=fig, data=data, super_title=title, save_loc=save_loc)
            # plt.show()

            # print update
            print('\r' + title + ': {:.2f}% complete'.format(100 * (i + 1) / len(image_files)), end='r')

        # new line
        print()


if __name__ == '__main__':

    # loop over the result folders
    for i in range(len(mdl_dirs)):

        # AE or VAE
        if mdl_dirs[i].find('VAE') >= 0:
            mdl_func = VariationalAutoEncoder
        else:
            mdl_func = AutoEncoder

        # load model parameters
        mdl_params = pickle.load(open(os.path.join(RESULT_DIR, mdl_dirs[i], 'mdl_params.p'), 'rb'))

        # build model
        mdl = mdl_func(input_dim=mdl_params['input_dim'],
                       latent_dim=mdl_params['latent_dim'],
                       conv_layers=mdl_params['conv_layers'],
                       full_layers=mdl_params['full_layers'],
                       lr=mdl_params['lr'],
                       px_z=mdl_params['px_z'],
                       batch_size=mdl_params['batch_size'],
                       n_epochs=mdl_params['n_epochs'],
                       save_dir=os.path.join(RESULT_DIR, mdl_dirs[i]))

        # make reconstruction directory
        recon_dir = os.path.join(RECON_DIR, mdl_dirs[i])
        if not os.path.exists(recon_dir):
            os.makedirs(recon_dir)

        # generate reconstructed images
        reconstruct_images(mdl, mdl_params, mdl_dirs[i])


