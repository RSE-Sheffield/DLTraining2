#!/bin/bash
#$ -l gpu=2 -P rse-training -q rse-training.q -l rmem=10G

module load libs/caffe/rc3/gcc-4.9.4-cuda-8.0-cudnn-5.1-conda-3.4-TESTING
source activate caffe
export LD_LIBRARY_PATH="/home/$USER/.conda/envs/caffe/lib:$LD_LIBRARY_PATH"

# (These example calls require you complete the LeNet / MNIST example first.)
# time LeNet training on CPU for 10 iterations
caffe time -model code/lab05/mnist_lenet.prototxt -iterations 10

# time LeNet training on GPU for the default 50 iterations
caffe time -model code/lab05/mnist_lenet.prototxt  -gpu 0

# time LeNet training on the first two GPUs for the default 50 iterations
caffe time -model=code/lab05/mnist_lenet.prototxt   -gpu 0,1

#If you want to run the code below, make sure you've trained the
#MNIST model first and are pointing to the correct .caffemodel file

# time a model architecture with the given weights on the first GPU for 10 iterations
#caffe time -model code/lab05/mnist_lenet.prototxt  -weights mnist_lenet_iter_10000.caffemodel -gpu 0 -iterations 10

# time a model architecture with the given weights on the first two GPUs for 10 iterations
#caffe time -model code/lab05/mnist_lenet.prototxt  -weights mnist_lenet_iter_10000.caffemodel -gpu 0,1 -iterations 10
