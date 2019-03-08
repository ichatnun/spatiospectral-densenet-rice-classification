# spatiospectral-densenet-rice-classification
Example data and Keras implementation of a deep convolutional neural network described in [ArXiv: Rice Classification Using Spatio-Spectral Deep Convolutional Neural Network](https://arxiv.org/abs/1805.11491).

Overview
------
A non-destructive rice variety classification system that benefits from the synergy between hyperspectral imaging and deep convolutional neural network (CNN) is developed. The proposed method uses a hyperspectral imaging system to simultaneously acquire complementary spatial and spectral information of rice seeds. The rice variety of each rice seed is then determined from the acquired spatio-spectral data using a deep CNN with hundreds of processing layers.

Files
------
* **script_run_proposed_deep_CNN.py** is the main file. 

* **utils_rice.py** contains the modules needed for the main file.

* **x.npy** contains example datacubes of the processed rice dataset that can be used for training/testing. Each datacube is a three-dimensional 50x170x110 tensor: two spatial dimensions and one spectral dimension.

* **labels.npy** contains the corresponding labels of the datacubes stored in **x.npy**



