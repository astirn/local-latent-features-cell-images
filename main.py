from cloud_run import cloud_run

# cloud run?
if cloud_run():
    import matplotlib as mpl
    mpl.use('Agg')

import os
from matplotlib import pyplot as plt
from utils import load_partitioned_data, split_data
from models import AutoEncoder, VariationalAutoEncoder, train, generate_latent_matrix

# cloud run?
if cloud_run():

    # load all partitioned data
    data = load_partitioned_data()

    # result directory
    RESULT_DIR = os.path.join(os.getcwd(), 'ResultsReal')

    # number of epochs
    N_EPOCHS = 250

# local run
else:

    # load some partitioned data
    data = load_partitioned_data(n_files=80)

    # result directory
    RESULT_DIR = os.path.join(os.getcwd(), 'ResultsTest')

    # number of epochs
    N_EPOCHS = 3

# split the data
i_train, i_valid = split_data(data.shape[0], 0.1)

# get input channels
n_channels = data.shape[-1]

# declare architectures
architectures = [{'conv_layers': [{'k_size': 9, 'out_chan': n_channels * 2},
                                  {'k_size': 6, 'out_chan': n_channels * 4},
                                  {'k_size': 3, 'out_chan': n_channels * 8}],
                  'full_layers': [3000, 2000]},
                 {'conv_layers': [{'k_size': 6, 'out_chan': n_channels * 2},
                                  {'k_size': 3, 'out_chan': n_channels * 4}],
                  'full_layers': [2000]}
                 ]

# loop over the architectures
for i in range(len(architectures)):

    # build AE model
    save_dir = os.path.join(RESULT_DIR, 'AE', 'Gauss', 'Arch{:d}'.format(i + 1))
    ae = AutoEncoder(input_dim=list(data.shape[1:]),
                     latent_dim=1000,
                     conv_layers=architectures[i]['conv_layers'],
                     full_layers=architectures[i]['full_layers'],
                     lr=1e-4,
                     px_z='Gaussian',
                     batch_size=100,
                     n_epochs=N_EPOCHS,
                     save_dir=save_dir)

    # train the model and generate latent space
    train(ae, data[i_train], data[i_valid])
    generate_latent_matrix(ae, data)

    # build AE model
    save_dir = os.path.join(RESULT_DIR, 'AE', 'Bern', 'Arch{:d}'.format(i + 1))
    ae = AutoEncoder(input_dim=list(data.shape[1:]),
                     latent_dim=1000,
                     conv_layers=architectures[i]['conv_layers'],
                     full_layers=architectures[i]['full_layers'],
                     lr=1e-4,
                     px_z='Bernoulli',
                     batch_size=100,
                     n_epochs=N_EPOCHS,
                     save_dir=save_dir)

    # train the model and generate latent space
    train(ae, data[i_train], data[i_valid])
    generate_latent_matrix(ae, data)

    # build VAE model
    save_dir = os.path.join(RESULT_DIR, 'VAE', 'Gauss', 'Arch{:d}'.format(i + 1))
    vae = VariationalAutoEncoder(input_dim=list(data.shape[1:]),
                                 latent_dim=1000,
                                 conv_layers=architectures[i]['conv_layers'],
                                 full_layers=architectures[i]['full_layers'],
                                 lr=1e-5,
                                 px_z='Gaussian',
                                 full_var=False,
                                 batch_size=100,
                                 n_epochs=N_EPOCHS,
                                 save_dir=save_dir)

    # train the model and generate latent space
    train(vae, data[i_train], data[i_valid])
    generate_latent_matrix(vae, data)

    # build VAE model
    save_dir = os.path.join(RESULT_DIR, 'VAE', 'Bern', 'Arch{:d}'.format(i + 1))
    vae = VariationalAutoEncoder(input_dim=list(data.shape[1:]),
                                 latent_dim=1000,
                                 conv_layers=architectures[i]['conv_layers'],
                                 full_layers=architectures[i]['full_layers'],
                                 lr=1e-4,
                                 px_z='Bernoulli',
                                 full_var=False,
                                 batch_size=100,
                                 n_epochs=N_EPOCHS,
                                 save_dir=save_dir)

    # train the model and generate latent space
    train(vae, data[i_train], data[i_valid])
    generate_latent_matrix(vae, data)

plt.ioff()
plt.show()
print('All done!')
