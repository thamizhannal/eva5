Modular Implementation
We have created a eva5-jedi project and exported it as open source https://test.pypi.org/project/eva5-jedi/0.0.1/

It consist of following modules

batchnorm.py: GBN implementation
dataloader.py: - train and test data loader
Engine_train_test.py: - train and test functions
model.py: Architecture creation and view
config.py: parameters needs to set to run the following main file. Will be adding further to config when notebook would be created
main.py: The final code which utilizes classes from other files. This is the code you would run on the final colab file
Target:
Reached 80% of validation accuracy in 20th epoch and it is consistent in next consecutive epochs too. We used GBN, Convolution, Depth wise separable convolution, Dilated convolution and Global average pooling in our network.

Results:
Parameters: 642,778
Best Train Accuracy: 82.53%
Best Test Accuracy: 82.25%
greater than 80% validation accuracy from 21-30th epochs.
Analysis:
Dropout added the necessary regularization required to make model robust
Also Image augumentation & GBN techniques helped to increase validation accuracy.
All the changes added to the base CNN architechture to acheive the target accuracy
Input image normalization
Image augmentation - rotation for Train data
Batch Normalization
Dropouts
Convolition, Depthwise separable convolution, Dilated convolution
LR scheduler
GAP
