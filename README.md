# Adaptive Segmentation Domain

This repository contains all the necessary tools to train segmentation algorithms in a domain adaptive fashion.

## Installation
Ensure you have [Python](https://www.python.org/) (we tested with Python 3.8) installed, and preferably a [CUDA capable device](https://developer.nvidia.com/cuda-gpus).

Clone this repository to a directory of your choice and install the required dependencies:
<pre><code>git clone https://github.com/unprompted2/adaptive-segmentation-domain
cd adaptive-segmentation-domain
pip install -r requirements.txt
</code></pre>

## Usage

### Illustrative examples
You can run an unsupervised domain adaptive demo by running the following command:
```
python train/train_unsupervised.py -c train/train_unsupervised.yaml
```
The provided YAML file is a configuration file that contains all necessary parameters. Note that you may have to adjust the data paths, depending on where you downloaded the data. By default, this will train a Y-Net with the EPFL and VNC data as source and target, respectively.

Similarly, you can run a semi-supervised domain adaptive demo by running the following command:
```
python train/train_semi_supervised.py -c train/train_semi_supervised.yaml
```
By default, this will do the same thing as the previous demo, but additionally employ labeled target data.

Feel free to adjust the parameter settings in the configuration files and examine the effect on the outcome!

### Domain adaptive training on your own data
To train a model in a domain adaptive fashion, you will need a (preferably large) labeled source datasets, e.g. one of the datasets provided in our repository. You will also need a target dataset that is at least partially labeled. These labels will be used for testing and evaluating performance. In the case of unsupervised domain adaptation, you are good to go. However, if the gap between the source and target is still relatively large, you are recommended to label a small part of the target data and use this for training (i.e. sem