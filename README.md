# local-latent-features-cell-images
Locally Latent Spatial Features of Cell Images with Multiple Biomarker Channels

## Repository Overview

### Files

- 'analysis.py'

- 'main.py'

- 'models.py'

- 'utils.py'

- 'run_mode.py'


### Folders

- 'Data'
- 'DataPartitions' 
- 'MNIST-data'
- 'Reconstruction'
- 'ResultsReal'
- 'ResultsReal'
- 'Test'

# Requirements

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