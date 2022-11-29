
## SAR Target Classification Using the Multikernel-Size Feature Fusion-Based Convolutional Neural Network
This is the unofficial implementation of MKSFF-CNN

Paper: https://ieeexplore.ieee.org/document/9528903 

## code
The backbone of the proposed MKSFF-CNN includes three convolutional layers, two pooling layers, two fully connected layers, and a softmax classifier. When the SAR image chip is input, the multi-kernel feature information of the input data is parallel extracted in the convolution layers, and the extracted multi-kernel features are fused in an optimal way to minimize the loss. The number of convolutional kernels with different sizes in each convolutional layer is set to three, and their sizes are set to three, five, and seven, respectively. Finally, the fused multi-kernel features of different dimensions are concatenated and fed into the first fully connected layers. The fully connected layers and the softmax classifier constitute the decision-making part.
