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
To train a model in a domain adaptive fashion, you will need a (preferably large) labeled source datasets, e.g. one of the datasets provided in our repository. You will also need a target dataset that is at least partially labeled. These labels will be used for testing and evaluating performance. In the case of unsupervised domain adaptation, you are good to go. However, if the gap between the source and target is still relatively large, you are recommended to label a small part of the target data and use this for training (i.e. semi-supervised DA).

To start training on your own data, you will first have to convert the format of your input data and labels. We currently support 3D volumes as PNG, JPG or TIF sequences, or multipage TIF data. To convert your data in given formats, consider using tools such as [Bio-Formats](https://www.openmicroscopy.org/bio-formats/).

Next, you will have to adjust the data settings in the configuration file. Make sure to correctly specify the training, validation, and testing data splits of both the source and target.

You should now be able to train on your own data, either unsupervised (with `train/train_unsupervised.py`) or semi-supervised (with `train/train_semi_supervised.py`).

### Parameter optimization
As like any other algorithms, parameter optimization plays a crucial role in domain adaptive segmentation. By running the following command, you can perform a grid search cross validation for a specific use-case:
```
python cross_validation/cross_validate.py -c cross_validate/cross_validate.yaml
```
This will perform 3-fold grid search cross validation w.r.t. the reconstruction regularization parameter of Y-Net for unsupervised domain adaptive training with the EPFL and VNC data as source and target, respectively. These settings can be adjusted in the configuration file. Note that this process can be computationally intensive, especially when the number of parameters in the grid or the amount of folds increases.

### Segmenting new datasets with pretrained models
Applying pretrained models on new datasets is straightforward. Make sure the new data is in the right format (see two sections before). You can segment the data by running the following command.
```
python test/segment.py -c test/segment.yaml
```
Ensure the configuration file specifies the correct data paths.

## FAQ
**Q: Is a CUDA capable device required?**
**A:** Technically, no. However, training on the CPU is very slow, so using a GPU is highly recommended. If you do not have a GPU at your disposal, consider using [Google Colab](https://colab.research.google.com/).

**Q: Is training with multiple source domains possible?**
**A:** This is called multi-domain adaptation and currently not supported.

**Q: Can I train with multiple GPUs?**
**A:** Yes. Our implementation leverages [PyTorch Lightning](https://www.pytorchlightning.ai/) which makes this relatively straightforward. You can specify your preferred compute GPUs and parallelization accelerator in the configuration file. Training on multiple nodes in cluster environments should also be possible with minor adjustments. For more details, we refer to the [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/) (e.g. [multi-GPU training](https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html) & [cluster computing](https://pytorch-lightning.readthedocs.io/en/latest/clouds/slurm.html)).

## Acknowledgements
Unprompted2 extends gratitude to the contributors of the [DAT](https://github.com/fungtion/DANN) and [MMD](https://github.com/OctoberChang/MMD-GAN) repositories that were used in this work.