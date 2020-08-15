### Problem Statement: ###

Refer to this code: COLABLINK https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx
WRITE IT AGAIN SUCH THAT IT ACHIEVES

- 99.4% validation accuracy

- Less than 20k Parameters

- You can use anything from above you want.

- Less than 20 Epochs

- No fully connected layer

- To learn how to add different things we covered in this session, you can refer to this code: https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99 DONT COPY ARCHITECTURE, JUST LEARN HOW TO INTEGRATE THINGS LIKE DROPOUT, BATCHNORM, ETC.

  

####  Approach & Architecture:   ###



**Input Data Set:** MNIST dataset is consist of human hand written monochrome images of size 28x28. Our objective is to detect human hand written digits using simple DNN.

**Network Architecture:** We chosen to experiment squeeze & expansion architecture to detect numbers in MINIST dataset. This architecture consist of convolution blocks followed by transition blocks.

**Convolution Block:** In convolution block, we use convolution layers that consist of kernels of size in incremental order e.g 8, 16, 20.

**Transition Block:** In transition block, we reduced num of kernel size to 8 from 20. After convolution block, we applied 1x1 convolution in transition block that helped us to reduce num of kernels. This is squeeze operation.

**Edges and Gradients:** Since input images are small in size, our network can expected extract edges and gradients at the Receptive Fields of 5-7. It is required to have 2-3 convolution layers to detect edges & gradients

**Max Pooling:** It filters out least important features and sends out most important features to consecutive layers for prediction.

**1x1 convolution**

**Batch Normalization** We applied batch normalization after every convolution(cond2d()). BN helped us to standardize the input to a convolution layer for every mini-batch. This stabilizes the learning process and also accelerates the DNN training.

**Avg Pooling** We applied Avg. Pooling in before prediction that calculates the average value for each pixel on the feature map.

**Early Stopping** If validation accuracy is not improving for more than 10 epoch we stopped trainings

**Train Vs Validation Plot** We captured train loss, train accuracy, validation loss & validation accuracies for all epochs and plotted a) train vs validation loss plot and b) train vs validation loss plots. These helps us to figure out gap between train vs validation loss & accuracies.



### **Train Vs validation Loss:** ###

![]( https://raw.githubusercontent.com/thamizhannal/eva5/master/MNIST_TrainVsValPlots.png )

### **Conclusion:**

I have implemented a simple convolution neural network architecture, that consist of two convolution and transition blocks (CT) with default batch size=32, epoch=20 and optimizer=SGD.

**First convolution Block:** Consist of 3 convolution layers of channel size 8, 16 & 20 **First Transition Block:** Consist of Max Pooling layer of 2x2 followed by 1x1 convolution of 8 channels. Here we reduced channels size from 20 to 8.

**Second Convolution Block:** Consist of 2 convolution layers of channel size 16 & 20

**Second Transition Block:** Consist of Average Pooling.

**Final Prediction:** Final convolution layer reduced channel size from 20 to 10 with 3x3 kernel and predicted digits.

### **Output:**

**We trained DNN consist of  10,370  parameters for 20 epoch and reached highest validation accuracy **
**99.49% at 11th epoch.**

**Train Vs validation accuracy plot:** It is evident that small gap between train & validation accuracies from 1st to 11th epochs, that means model learns better, after that gap between train vs validation increases that seems model is not learning much and we have to add additional methods to make it better.

**Train Vs validation loss plot:** From 1st to 11th epoch train loss is start reducing drastically and gradually. There was less gap between train vs validation loss means that **model is not overfitting,** after 11th epoch loss between train vs validation start increasing. So, it is better to stop at 11th epoch after which model does not improve much or we have to try additional approaches to improve this model.

