## Installation Guide

### Install Dependencies:

1. Install Ceres: http://ceres-solver.org/installation.html
2. Ceres HeLLo World: http://ceres-solver.org/nnls_tutorial.html#hello-world
3. Install glog: See how to use glog in with CMAKE - Do not install from source. Installing from souce gives more flexibility with CMake. apt install has missing files**.

        sudo apt install libgoogle-glog-dev

4. Install boost:

        sudo apt-get install libboost-all-dev

### Install Package

1. mkdir build && cd build
2. cmake ..
3. make

### Datasets to Experiment With
1. https://lucacarlone.mit.edu/datasets/
