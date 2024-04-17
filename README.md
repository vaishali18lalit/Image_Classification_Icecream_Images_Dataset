# Image Classification with AlexNet
# Overview
This project focuses on image classification using the AlexNet architecture. The goal is to classify images into predefined categories using deep learning techniques. The project utilizes the PyTorch framework and torchvision library for model implementation and training.

# Project Details
## Model Architecture
### AlexNet:
Pretrained model imported from the torchvision library.
Modified the last fully connected layer to adapt to the number of classes in the dataset.
## Data Augmentation
Implemented data augmentation techniques to enhance the diversity and robustness of the dataset.
Techniques applied include random rotation, color jittering, random affine transformations, and Gaussian blurring.
## Training and Evaluation
Split the dataset into training, validation, and test sets.
Utilized DataLoader for batching and shuffling the data.
Trained the model using the training set and evaluated its performance on the validation set.
Fine-tuned the model parameters and hyperparameters based on validation performance.
Evaluated the final model on the test set to assess its generalization performance.# Image_Classification_Icecream_Images_Dataset
Image Classification performed on ICECREAM dataset using ALEXNET pretrained model

## Results

Validation Performance:
Validation Accuracy: 100%
Test Performance:
Test Accuracy: 94.32%
