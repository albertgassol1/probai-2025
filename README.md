# ProbAI 2025 - Flow Matching in MNIST

## Introduction

This repository implements conditional flow matching in the MNIST dataset for the ProbAI 2025 summer school. Specifically, it covers three configurations:

- Gaussian source distribution.
- MNIST digit with crop noise, matched with random target digit.
- MNIST digit with crop noise, matched with its unaltered version.

## Installation

To set up the environment and install dependencies (tested on python 3.8.10):

```bash
python -m venv probai
source probai/bin/activate
# Install required packages
pip install -r requirements.txt
```

**Note:** The code was tested on an Ubuntu 20.04 machine with an NVIDIA RTX 2000 Ada Generation Graphics Card GPU running CUDA 12.1.

## Data
Download the MNIST dataset using the following python script.

```
python dataloader/download_mnist.py --data_path <directory where you want to store the MNIST dataset>
```

## Train

A single trainer script is used to train all the configurations. First, specify the parameters and hyperparameters in the [yaml config files](./config/). It is important to specify the output path, data path, checkpoint parh (if resuming training), and wether you want to use W&B logging.

 Then run the following commands.

 **Gaussian source distribution:**
 ```
 python trainer.py config_name=gaussian_prior
 ```

  **MNIST digit with crop noise, matched with random target digit:**
 ```
 python trainer.py config_name=mnist_crop_shuffle_prior
 ```

   **MNIST digit with crop noise, matched with its unaltered version:**
 ```
 python trainer.py config_name=mnist_crop_prior
 ```


## Evaluation

A single script is used to generate the plots of the report. You can run it using the following commands.


 **Gaussian source distribution:**
 ```
python generate_random.py --model_path <checkpoint path> --save_dir <path where you want to save the plots> --config_path config/gaussian_prior.yaml
 ```

  **MNIST digit with crop noise, matched with random target digit:**
 ```
python generate_random.py --model_path <checkpoint path> --save_dir <path where you want to save the plots> --config_path config/mnist_crop_shuffle_prior.yaml
 ```

   **MNIST digit with crop noise, matched with its unaltered version:**
 ```
 python generate_random.py --model_path <checkpoint path> --save_dir <path where you want to save the plots> --config_path config/mnist_crop_prior.yaml
 ```
