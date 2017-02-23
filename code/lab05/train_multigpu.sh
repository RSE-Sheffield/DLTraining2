#!/bin/bash
#$ -l gpu=2 -P rse-training -q rse-training.q -l rmem=10G -j y

module load libs/caffe/rc3/gcc-4.9.4-cuda-8.0-cudnn-5.1-conda-3.4-TESTING
source activate caffe
export LD_LIBRARY_PATH="/home/$USER/.conda/envs/caffe/lib:$LD_LIBRARY_PATH"

caffe train -solver=code/lab05/mnist_lenet_solver.prototxt -gpu=0,1