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
The provided YAML file is a configuration file that contains all necessary parameters. Note that you may have to adjust the data paths, depending on where you downloaded the data. By default, this will train