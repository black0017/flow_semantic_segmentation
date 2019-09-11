# Flownet2 with semantic segmentation on KITTI15 dataset

#### Based on original Pytorch implementation of [FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925). available at {https://github.com/NVIDIA/flownet2-pytorch}


## Our contribution in the loss function
Core Idea: The major errors in optical flow estimation appear to the pixels located on (or close to) the object outline. These boundary regions are mainly responsible for higher errors thus they should have
more significant role in the optical flow estimation.

Thus, we proposed to generate a loss term which will focus on the outline of each instance (binary change detection mask).
Object outlines will be estimated by the *change* of the segmentation masks instance on the image pairs.
Finally, element-wise multiplication between the estimated optical flow and the change detection mask will be applied. Both matrices should have the same spatial dimensions.(same as the input image pairs)
It should be noticed that the element-wise multiplication can be integrated into the loss function of any optical flow network.

In order to use the semantic segmentation masks we used Pyramid scene parsing network(PSP-net) which has been proposed by  Zhao et al.
The network has been fine-tuned on KITTI15 dataset using the 200 annotated segmentation images. The output of the segmentation model is the segmentation mask for each RGB input image as depicted below.
Segmentation masks were generated offline.

## Implementation Details
We nearly reproduced the original results of Flownet2S, as a baseline method trained only on KITTI15 dataset. Dataloaders for KITTI15 are provided.
We trained the network for 500 epochs with 160 training and 40 validation image pairs (official split).
We used static random crop (320,1218) in order to have the inputs in a specific shape.
Batch size was set to 32. Learning rate was set to 0.0001 with Adam optimizer and Multiscale L1 as a loss function.
The configuration was the same for both experimental setups in order to observe significant results due to the integration of our mask in the loss function.
Average end-point error(AEE) metric that was used to evaluate the results of the predicted flow using the  ground truth.
This metric calculates the average end-point error (epe), taking into account only the valid pixels, as they have been officially annotated.

## Vanilla Flownet supported Network architectures
Below are the different flownet neural network architectures that are provided based on the original repo. For computational complexity issues we used the FlowNet2S which is the lighter one. Pretrained models are also available but we did not used them.

 - **FlowNet2S**
 - **FlowNet2C**
 - **FlowNet2CS**
 - **FlowNet2CSS**
 - **FlowNet2SD**
 - **FlowNet2**

## Build and launch Docker image
Libraries and other dependencies for this project include: Ubuntu 16.04, Python 2.7, Pytorch 0.2, CUDNN 6.0, CUDA 8.0.

A Dockerfile with all the above dependencies is available as in the original repo : <br />

    bash launch_docker.sh

## Reference 
Original Flownet paper
````
@InProceedings{IMKDB17,
  author       = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
  title        = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
  booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
  month        = "Jul",
  year         = "2017",
  url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
}
````
```
@misc{flownet2-pytorch,
  author = {Fitsum Reda and Robert Pottorff and Jon Barker and Bryan Catanzaro},
  title = {flownet2-pytorch: Pytorch implementation of FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NVIDIA/flownet2-pytorch}}
}
```
## Acknowledgments
Parts of this code were derived from [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch). 
