# local-latent-features-cell-images
#### Locally Latent Spatial Features of Cell Images with Multiple Biomarker Channels
This work processes multi-channel lung cancer cell images to generate a compact latent space for local cellular regions.
At a high level, latent space discovery is an important first step to reducing these images to a compact vector
representation from which researchers can:
- By conditioning the latent space on treatment outcome, generate cellular images to qualitatively identify
characteristics linked to treatment success.
- Use the latent space to infer treatment outcome for future patients. 

## Repository Overview

### Files

- 'analysis.py'--Run this file after 'main.py' to generate full-size .png image reproductions for all image files in the
'Data' folder. Specifically, this files employs the trained models resulting from 'main.py' to generate reconstructions
for the image partitions and thereafter stitch them together into a full-size format.

- 'main.py'--Run this file after 'utils.py'. This file trains the two model families for all the listed architectures.
If 'test_mode()' returns False, this file will load the entire data set and run for the full number of epochs.
If 'test_mode()' returns True, this file will load a data subset and run only for a few of epochs for code testing
purposes.

- 'models.py'--This file contains the Auto-Encoder (AE) and Variational Auto-Encoder (VAE) Class implementations in
TensorFlow. Additionally, it contains functions shared by these models to accomplish training, testing, latent-space
generation, and image reconstructions. This file can also be run directly. If so, it will download the MNIST data set
automatically and train four models: Bernoulli AE, Gaussian AE, Bernoulli VAE, and Gaussian VAE. The MNIST data set was
used to expedite development.

- 'utils.py'--This file **MUST BE RUN FIRST**. It processes the .tiff files in the 'Data' folder into 64x64x7 image
partitions and saves the results as .npy files in 'DataPartitions', which 'main.py' requires to run!

- 'test_mode.py'--This file has a single function that determines the operation mode of the above files. Note it will
load a terminal-friendly 'matplotlib' for cloud computing usage.


### Directories

- 'Data'--This is the original data directory. Inside there should be subject directories that each contain any number
of a subject's .tiff files. **Data is proprietary and therefore not available to public via this repository.**

- 'DataPartitions'--This directory is generated by 'utils.py'. It will contain a .npy file corresponding to each .tiff
image found by recursively searching the 'Data' directory.

- 'MNIST-data'--This directory is generated by 'models.py', which if run will download the MNIST data set to this
directory.

- 'Reconstruction'--This directory is generated by 'analysis.py'. It will contain two directories: 'AE' and 'VAE'
corresponding to the two model families employed. Within each, there will two more directories: 'Bern' and 'Gauss'
corresponding to the data treatment (Bernoulli and Gaussian). Within in each of these directories will be a number of
architectures folders corresponding to the declared architectures in 'main.py'. Within each architecture there will be
full-size .png reconstructions for every image partition file in 'DataPartitions'.

- 'ResultsReal'--This directory has the same hierarchy as 'Reconstruction'. However, it instead will contain TensorFlow
checkpoint and TensorBoard files generated by 'main.py', when 'test_mode()' returns False. Additionally, there
will be a 'Learning_Curve.png' and a series of images titled, 'Model_Performance_Epoch_{d}.png'. This image series shows
the reconstruction of a randomly selected image partition at the end of each training epoch.

- 'ResultsTest'--This directory has identical structure and form as 'ResultsReal', but is updated when 'test_mode()'
returns True.

- 'Test'--This directory has similar, but slightly different, structure as 'ResultsReal' and 'ResultsTest'. It is
generated when calling 'models.py' and will contain results for MNIST data.

## Requirements

- Python >= 3.5
- Python package requirements are listed in 'requirements.txt'. If you lack a CUDA capable graphics card, please change
    'tensorflow-gpu==1.4.1' to 'tensorflow==1.4.1'.
- System requirements:
    - Test Mode ('test_mode()' returns True): Can be run on a modern laptop with 16GB of RAM.
    - Real Mode ('test_mode()' returns False): Requires massive computational resources that scale with the amount of
        data. Original implementation processed approximately 400 1000x1000x7 multi-channel images. To load the full
        data set into RAM required a 'n1-highmen-16' instance on the Google Cloud Platform with an NVIDIA Tesla P100
        GPU. This configuration amounts to 16 vCPUs and 104 GB of RAM. Code refactoring could reduce this requirement
        by separately loading data for each batch from the 'DataPartitions' folder. However, this high-frequency
        loading would likely increase training time.