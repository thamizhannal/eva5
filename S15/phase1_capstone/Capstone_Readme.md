### Monocular Depth Estimation and Object Detection using Encoder-Decoder architecture

#### Problem Description:

1. Your assignment is to create a network that can perform 3 tasks simultaneously:
   1. Predict the boots, PPE, hardhat, and mask if there is an image
   2. Predict the depth map of the image
   3. Predict the Planar Surfaces in the region
2. As always you'll find in the industry, people don't really share their datasets. Moreover, datasets can be really HUGE and very difficult to handle. The strategy we will use here is to:
   1. use pre-trained networks and use their outputs as the ground truth data
      1. Use the output or MidasNet for the depth map
      2. Use the output of PlanerRCNN for the plane-surface data
   2. Use the dataset EVA5 batch collected for boots/PPE/mask/hardhat. 

### Introduction:

Objective of this work is to create a deep convolution neural network using transfer learning technique that performs image depth estimation and object detection using Yolo on custom images that has boots, PPE, hardhat, and mask objects.

Here, image depth is predicted using [MiDas](https://github.com/intel-isl/MiDaS) Network that consist of resnet101 layers and object detection is performed using standard [YoloV3](https://github.com/theschoolofai/YoloV3) code.  This DNN consist of resnet101 encoder and two decoders one for depth prediction and second one for object detection using [YoloV3](https://github.com/theschoolofai/YoloV3) .

https://github.com/NVlabs/planercnn

https://github.com/intel-isl/MiDaS

Yolo Input Size: 416x416

416, 208, 104, 52, 26, 13x13

FeatureFusionBlock: 

**Input Data Set:**

Yolov3 annotated dataset location in gdrive: https://drive.google.com/drive/u/0/folders/1wPxFaaus2FQZ7aMoBif1eqtxW4mcCUfj

**custom data description:**

This the data of images regarding the PPE kit, primarily for construction workers. It has 4 classes which are **`hardhat, vest, mask, boots`**. Below is the definition of the all the files in the shared google drive

**Total number of Images:** `3489`

| file name                 | description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| images                    | Input images that are untouched                              |
| labels                    | contains 1 .txt file per image with information of class cx(bounding box centre) cy(bounding box centre) h(ratio to reduce anchor box height) w(ratio to reduce anchor box width) |
| train.txt, test.txt       | text files with location of training and text images respectively |
| train.shapes, test.shapes | text files with shapes of training and text images respectively |
| custom.data               | File describing the number of classes and local to train.txt and test.txt |
| custom.names              | text file with class label descriptions                      |
| midas_out_colormap        | Images which have depth information of every image from the images folder on a colourful scale. Implemented from the repo: [MiDas](https://github.com/intel-isl/MiDaS) using the colab file: [assign_14a](https://github.com/theschoolof-ai/JEDI/blob/master/S14/Session14_MiDas.ipynb) |
| midas_out_greyscale       | Images which have depth information of every image from the images folder on a grey scale. Implemented from the repo: [MiDas](https://github.com/intel-isl/MiDaS) using the colab file: [assign_14a](https://github.com/theschoolof-ai/JEDI/blob/master/S14/Session14_MiDas.ipynb) |
| plane_rcnn_inference      | Images which have plane segmentation information of every image from the images folder. PlaneR-CNN, that detects arbitrary number of planes, and reconstructs piecewise planar surfaces from a single RGB image was implemented from the repo: [planercnn](https://github.com/NVlabs/planercnn) using the following colab file: [assign_14b](https://github.com/theschoolof-ai/JEDI/blob/master/S14/assignment_EVA5_JEDI_14b.ipynb) |



**Architecture**:

**Encoder:**  resnet101 network with Midas pretrained weights . This is common across multiple decoders such as depth detection, object detection & planercnn. As of now depth and object detection has been implemented. Encoder outputs 4 layers of convolution feature maps such as layer1, layer2, layer3 and layer4.

1. layer1: img_size/4
2. layer2 img_size/8
3. layer3: img_size/16
4. layer4: img_size/32

**Decoder**:

**Midas**: Decoder consist of 4 layers of Feature Fusion Blocks each of two Residual Convolution Units as in Midas implementation. Encoder output is 4 layered convolution feature blocks, they were up sampled and connected to respective Feature Fusion Blocks in Midas decoder.

**Yolov3:** Instead if Darknet encoder, we used resnet101 encoder here. Yolo decoder has three outputs which are of the resolution of img_size/32, img_size/16, img_size/8 and are predicted at an anchor box and channel level which in this case is 9, 3 respectively. 

<img src="C:\Users\tparamas\AppData\Roaming\Typora\typora-user-images\image-20201206204513755.png" alt="image-20201206204513755" style="zoom:50%;" />



YoLoV3 Decoder: This consist of 3 layer prediction of anchor boxes of size 13, 26, 52. These were created using standard YoloV3 decoders.

![image-20201206204544294](C:\Users\tparamas\AppData\Roaming\Typora\typora-user-images\image-20201206204544294.png)

### Model Training:

During model training, we preloaded MiDas weights from resnet101. So encoder layers were frozen and trained only decoders.



#### Yolo Model Training command:

YoloV3: It uses Darknet-53 as the backbone network and uses three scale predictions for anchor boxes  such as 13, 26 & 52 respectively.

!python fork_train.py --data /content/gdrive/'My Drive'/SchoolOfAI_EVA/YoloV3_S13/YoloV3/data/customdata/custom.data --batch 5 --cache --epochs 15 --nosave --train_decoder=yolo

cfg- configuration file

data- custom dataset location



#### Depth & Yolo Model training:

!python fork_train.py --cfg /content/gdrive/'My Drive'/SchoolOfAI_EVA/phase1_capstone/cfg_yolo/yolov3-custom.cfg --data /content/gdrive/'My Drive'/SchoolOfAI_EVA/YoloV3_S13/YoloV3/data/customdata/custom.data --batch 5 --cache --epochs 15 --nosave --train_decoder=all



### Loss Function:

For Yolo, we used standard loss function defined in YoloV3 architecture

For Depth, we used MSE(Mean squared Error) and SSIM (structural similarity index measure)



```
mse_loss = nn.MSELoss()
ssim_loss = SSIM()
depth_mseloss = mse_loss(depth_pred.unsqueeze(0).permute(1, 0, 2, 3),
                         depth_targets.unsqueeze(0).permute(1, 0, 2, 3))
depth_ssimloss = 1 - ssim_loss(depth_pred.unsqueeze(0).permute(1, 0, 2, 3),
                               depth_targets.unsqueeze(0).permute(1, 0, 2, 3))
depth_loss = _mseloss * depth_mseloss + _ssimloss * depth_ssimloss
dl_epoch += depth_loss

# Compute yolo loss
yolo_loss, loss_items = compute_loss(yolo_pred, targets, model)
yolo_loss = _yololoss * yolo_loss
yl_epoch += yolo_loss

# final loss
final_loss = depth_loss + _yololoss * yolo_loss
fl_epoch += final_loss
```

### Tensor Board:



Yolo Loss : Step Vs Value for single image. This image shows as step progress loss value reduces from 130 to 80.

![image-20201206192328703](C:\Users\tparamas\AppData\Roaming\Typora\typora-user-images\image-20201206192328703.png)

##### Yolo mAP loss: for 14 epochs 

mAP: mean Average Precision

![image-20201206193555033](C:\Users\tparamas\AppData\Roaming\Typora\typora-user-images\image-20201206193555033.png)



| Script                    | description                                          |
| ------------------------- | ---------------------------------------------------- |
| model.py                  | encoder & Decoder code                               |
| model_yolo.py             | yolov3 decoder from base yolov3 code                 |
| train_fork.py and test.py | Train and evaluation of the model during Training    |
| util_yolo                 | All the util functions like parsing and loading data |
| blocks.py                 | utility functions for building Depth decoder         |
| depth_loss                | SSIM function for evaluation for depth loss          |

#### Conclusion:

Deep neural network using Encoder and Decoder architecture has been created using resnet101 as single encoder and customized Midas net and Yolov3 network as decoder . Applied transfer learning to load pretrained weights. 

First we trained YoloV3 model for 14 epochs and frozen Depth estimation decoder , it has been showed that Yolo loss was reduced.

Second we trained both depth & Yolo Model for 14 epochs and it has been showed that depth loss was reduced.



