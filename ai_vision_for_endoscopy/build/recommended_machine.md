To run this project's code successfully, it must be run in an appropriate machine with an appropriate environment.

To make success more likely or easier (with less trouble-shooting), I suggest using a machine such as an AWS EC2 instance with the following pre-configured environment and of the following type:

* Instance Environment: Deep Learning AMI (Ubuntu) Version 23.0 - ami-058f26d848e91a4e8
** Includes: MXNet-1.4, TensorFlow-1.13, PyTorch-1.1, Keras-2.2, Chainer-5.4, Caffe/2-0.8, Theano-1.0 & CNTK-2.7, configured with NVIDIA CUDA, cuDNN, NCCL, Intel MKL-DNN, Docker & NVIDIA-Docker.
* Instance Type: GPU instance: p2.xlarge  (vCPUs: 4; Memory: 61 GiB)

