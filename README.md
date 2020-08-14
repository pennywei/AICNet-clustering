Requirements：
Intel i7-7700 CPU, Nvidia GeForce GTX 1060 6GB; 
Python 3.6; 
Keras 2.2.0; 
Tensorflow 1.6.0; 
Win 10

Usage：
DCAE_VGG are first trained and the model weights are saved in spectralnet4k/src/pretrain_weights. To run AICNet on the 4k auroral images, please run the python file run.py in spectralnet4k/src/applications/run.py. Please modify line 84 and lines 210-211 in spectralnet4k/src/core/data.py to give the right paths of auroral data and DCAE_VGG model respectively.
More details about SpectralNet are available at https://github.com/KlugerLab/SpectralNet
