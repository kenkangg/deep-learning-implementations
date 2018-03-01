# Neural Style Transfer

An implementation of [neural style transfer](https://arxiv.org/pdf/1508.06576v2.pdf) in TensorFlow.

In order to reduce training time, this implementation is intended to be run on Google Colaboratory due to free GPU access


## Running

Open this notebook through Google Colaboratory

Upload the VGG19 weights onto a Google Cloud Platform Storage bucket and change "project-id" and "bucket name" variables accordingly.

Input your content/style images

## Requirements

### Pretrained Model

* [Pre-trained VGG19 network](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat) - Place this is a Google Cloud Platform Storage bucket in order to quickly load this onto the iPython notebook.

### Dependencies

* [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html#download-and-setup)
* [NumPy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)
* [SciPy](https://github.com/scipy/scipy/blob/master/INSTALL.rst.txt)
* [Pillow](http://pillow.readthedocs.io/en/3.3.x/installation.html#installation)
